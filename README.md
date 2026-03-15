# AI-sign-language-avatar

## Phase 1 (post-download) pipeline

After downloading WLASL videos into `WLASL-master/start_kit/raw_videos`, build one
normalized+smoothing JSON motion clip per gloss.

MediaPipe Holistic compatibility note: use Python 3.10-3.12. Some Python 3.13
builds of `mediapipe` do not expose `mp.solutions.holistic.Holistic`.

### 1) Install Python dependencies

```powershell
pip install mediapipe opencv-python numpy scipy
```

### 2) Build the motion library

Run from repository root:

```powershell
python data_pipeline/batch_process.py `
	--index WLASL-master/start_kit/WLASL_v0.3.json `
	--raw-videos WLASL-master/start_kit/raw_videos `
	--out-dir backend/motion_library `
	--max-per-gloss 5
```

Outputs:
- `backend/motion_library/*.json`: one selected clip per gloss.
- `backend/motion_library/report.json`: summary of written/skipped glosses.

### 3) Quick dry run (no file writes)

```powershell
python data_pipeline/batch_process.py --dry-run
```