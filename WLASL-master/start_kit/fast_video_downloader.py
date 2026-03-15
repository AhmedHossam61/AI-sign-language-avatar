import argparse
import json
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faster WLASL downloader with parallel workers.")
    parser.add_argument("--index", default="WLASL_v0.3.json", help="Path to WLASL json index")
    parser.add_argument("--out", default="raw_videos", help="Output directory")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--retries", type=int, default=3, help="Retry count per video")
    parser.add_argument("--insecure", action="store_true", help="Disable SSL certificate verification for problematic hosts")
    parser.add_argument("--youtube-only", action="store_true", help="Download only YouTube links")
    parser.add_argument("--nonyoutube-only", action="store_true", help="Download only non-YouTube links")
    # Subset options for fast testing / project bootstrapping
    parser.add_argument("--top-n", type=int, default=None, metavar="N",
                        help="Only download the first N glosses (words) from the index. "
                             "Useful for quickly bootstrapping the motion library (e.g. --top-n 100).")
    parser.add_argument("--words", nargs="+", metavar="WORD",
                        help="Whitelist of specific gloss words to download (case-insensitive). "
                             "E.g. --words hello book water")
    parser.add_argument("--max-per-gloss", type=int, default=None, metavar="M",
                        help="Maximum number of video instances to download per gloss. "
                             "Set to 1-3 for fast testing (e.g. --max-per-gloss 2).")
    return parser.parse_args()


def is_youtube(url: str) -> bool:
    u = url.lower()
    return "youtube.com" in u or "youtu.be" in u


def load_instances(index_path: Path, top_n: int = None, words: list = None, max_per_gloss: int = None):
    with index_path.open("r", encoding="utf-8") as f:
        content = json.load(f)

    # Filter by whitelist of words first
    if words:
        words_lower = {w.lower() for w in words}
        content = [e for e in content if e.get("gloss", "").lower() in words_lower]

    # Limit to first N glosses
    if top_n is not None:
        content = content[:top_n]

    instances = []
    for entry in content:
        entry_instances = entry.get("instances", [])
        if max_per_gloss is not None:
            entry_instances = entry_instances[:max_per_gloss]
        for inst in entry_instances:
            instances.append(inst)
    return instances


def request_video(url: str, referer: str = "", insecure: bool = False) -> bytes:
    user_agent = "Mozilla/5.0"
    headers = {"User-Agent": user_agent}
    if referer:
        headers["Referer"] = referer
    request = urllib.request.Request(url, None, headers)
    context = None
    if insecure:
        context = ssl._create_unverified_context()

    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        return response.read()


def download_nonyoutube(inst: dict, out_dir: Path, retries: int, insecure: bool) -> tuple[bool, str]:
    url = inst["url"]
    vid = inst["video_id"]

    if "aslpro" in url:
        save_to = out_dir / f"{vid}.swf"
        referer = "http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi"
    else:
        save_to = out_dir / f"{vid}.mp4"
        referer = ""

    if save_to.exists():
        return True, f"skip {vid}"

    last_err = None
    
    for attempt in range(retries):
        try:
            data = request_video(url, referer=referer, insecure=insecure)
            with save_to.open("wb") as f:
                f.write(data)
            return True, f"ok {vid}"
        except urllib.error.HTTPError as e:
            # 404 and 403 are permanent failures—don't retry
            if e.code in (404, 403):
                return False, f"fail {vid} HTTP {e.code} (permanent)"
            last_err = e
        except Exception as e:
            # Some hosts in WLASL have broken cert chains/hostnames.
            # Retry once with disabled verification if user did not enable --insecure.
            if not insecure and isinstance(e, ssl.SSLCertVerificationError):
                try:
                    data = request_video(url, referer=referer, insecure=True)
                    with save_to.open("wb") as f:
                        f.write(data)
                    return True, f"ok {vid} (ssl-fallback)"
                except Exception as fallback_err:
                    last_err = fallback_err
                    time.sleep(0.5)
                    continue

            last_err = e

        # Only add delay after first retry (not on final attempt)
        if attempt < retries - 1:
            time.sleep(0.5)

    return False, f"fail {vid} {last_err}"


def download_youtube(inst: dict, out_dir: Path, retries: int, insecure: bool) -> tuple[bool, str]:
    url = inst["url"]
    video_id = url[-11:]

    if (out_dir / f"{video_id}.mp4").exists() or (out_dir / f"{video_id}.mkv").exists() or (out_dir / f"{video_id}.webm").exists():
        return True, f"skip {video_id}"

    # --concurrent-fragments accelerates segmented streams.
    # --download-archive avoids re-downloading already completed URLs.
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-warnings",
        "--continue",
        "--retries",
        str(retries),
        "--fragment-retries",
        str(retries),
        "--concurrent-fragments",
        "4",
        "--download-archive",
        str(out_dir / "yt_archive.txt"),
        "-o",
        str(out_dir / "%(id)s.%(ext)s"),
        url,
    ]

    if insecure:
        cmd.insert(1, "--no-check-certificates")

    try:
        rv = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if rv.returncode == 0:
            return True, f"ok {video_id}"
        return False, f"fail {video_id} exit={rv.returncode}"
    except Exception as e:
        return False, f"fail {video_id} {e}"


def main() -> None:
    args = parse_args()

    if args.youtube_only and args.nonyoutube_only:
        raise SystemExit("Choose only one of --youtube-only or --nonyoutube-only")

    index_path = Path(args.index)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    instances = load_instances(index_path, top_n=args.top_n, words=args.words, max_per_gloss=args.max_per_gloss)

    if args.youtube_only:
        instances = [i for i in instances if is_youtube(i["url"])]
    elif args.nonyoutube_only:
        instances = [i for i in instances if not is_youtube(i["url"])]

    total = len(instances)
    print(f"Total queued: {total}")

    ok = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = []
        for inst in instances:
            if is_youtube(inst["url"]):
                futures.append(pool.submit(download_youtube, inst, out_dir, args.retries, args.insecure))
            else:
                futures.append(pool.submit(download_nonyoutube, inst, out_dir, args.retries, args.insecure))

        for idx, fut in enumerate(as_completed(futures), start=1):
            success, msg = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
            if idx % 50 == 0 or not success:
                print(f"[{idx}/{total}] ok={ok} fail={fail} :: {msg}")

    print(f"Done. ok={ok} fail={fail} total={total}")


if __name__ == "__main__":
    main()
