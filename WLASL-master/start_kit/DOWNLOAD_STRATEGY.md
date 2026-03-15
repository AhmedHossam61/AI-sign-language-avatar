# WLASL Download Strategy

## Problem Summary
- **21,083 total videos** in WLASL_v0.3.json
- Many videos are **dead links** (404 Not Found, 403 Forbidden)
- Many videos **timeout** (host not responding or very slow)
- Previous run: Only ~5 videos succeeded out of first 26 attempts

## Error Breakdown (from your last run)
| Type | Count | Meaning |
|------|-------|---------|
| `[WinError 2]` | Many | yt-dlp not installed (NOW FIXED) |
| `WinError 10060` | Many | Connection timeout (host is slow/dead) |
| `HTTP 404` | Many | URL permanently deleted |
| `HTTP 403` | Some | Access denied by host |
| `SSL cert error` | Few | Certificate issue (handled with --insecure) |

## Fast-Testing / Project Bootstrap (Recommended First Step)

Download a small subset to establish the project structure quickly:

```powershell
cd WLASL-master\start_kit
# Option A: Top 100 glosses, 2 videos each, non-YouTube only
python fast_video_downloader.py --index WLASL_v0.3.json --out raw_videos --workers 8 --retries 1 --nonyoutube-only --insecure --top-n 100 --max-per-gloss 2

# Option B: Specific words you need for MVP
python fast_video_downloader.py --index WLASL_v0.3.json --out raw_videos --workers 8 --retries 1 --nonyoutube-only --insecure --words hello book water help yes no please thank

# Option C: Top 50 words, 1 video each (fastest possible test)
python fast_video_downloader.py --index WLASL_v0.3.json --out raw_videos --workers 8 --retries 1 --nonyoutube-only --insecure --top-n 50 --max-per-gloss 1
```

**New subset flags:**
| Flag | Description |
|------|-------------|
| `--top-n N` | Only download the first N glosses (words) from the JSON |
| `--words W1 W2 ...` | Whitelist specific words by name (case-insensitive) |
| `--max-per-gloss M` | Cap video instances per word (e.g. `2` means 2 videos per sign) |

These flags combine freely with `--youtube-only` / `--nonyoutube-only`.

## Full Dataset Download Strategy

### Step 1: Download Non-YouTube Videos Only (Fastest & Most Reliable)
```powershell
cd WLASL-master\start_kit
python fast_video_downloader.py --index WLASL_v0.3.json --out raw_videos --workers 8 --retries 1 --nonyoutube-only --insecure
```

**Why this first:**
- Non-YouTube sources (aslpro, etc.) are more stable
- Only 1 retry needed (dead links fail fast now)
- `--insecure` handles broken SSL certs
- Saves time by avoiding YouTube timeouts
- You'll get a solid baseline of available videos

### Step 2: YouTube Videos (After you have a good local set)
```powershell
python fast_video_downloader.py --index WLASL_v0.3.json --out raw_videos --workers 4 --retries 3 --youtube-only
```

**Why separate:**
- YouTube downloads are separate infrastructure (yt-dlp)
- YouTube videos have different timeout behavior
- Fewer workers = less strain on network

## Expected Results
- **First run (non-YouTube)**: Expect 30-50% success rate (many old links are dead)
- **Total successful videos**: ~1000-3000 out of 21,083 are likely still available
- **Time estimate**: 1-3 hours depending on your network and the number of available videos

## Tips
1. Check the log file `download_TIMESTAMP.log` to see which videos failed
2. Dead videos won't be retried again (skipped if file already attempted in same run)
3. If you want to focus on only common signs for MVP, use `--top-n 100 --max-per-gloss 2 --nonyoutube-only` first, then manually curate good videos
4. You can safely interrupt (Ctrl+C) and resume—already downloaded files are automatically skipped
