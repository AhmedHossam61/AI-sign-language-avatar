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

## Recommended Download Strategy

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
3. If you want to focus on only common signs for MVP, use `--nonyoutube-only` first, then manually curate good videos
4. You can safely interrupt (Ctrl+C) and resume—already downloaded files are automatically skipped
