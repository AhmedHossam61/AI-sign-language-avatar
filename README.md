# AI Sign Language Avatar — Technical Implementation Plan

## Overview

A real-time system that converts English text into ASL (American Sign Language) sign animations played by a 3D avatar. The system targets **low latency** by avoiding generative/diffusion models in favor of a **skeleton-based animation library** built from real signer recordings.

---

## Full System Architecture

```mermaid
flowchart TD
    subgraph Input ["Input Layer"]
        A1[User Text Input]
        A2[Voice Input\nWhisper ASR — Phase 3]
    end

    subgraph NLP ["NLP Layer — Backend"]
        B[Text Preprocessor\nspaCy tokenizer + lemmatizer]
        C[ASL Gloss Converter\nrule-based MVP → Seq2Seq Phase 2]
        D[Gloss Tokenizer\nlemma → UPPERCASE tokens]
    end

    subgraph Animation ["Animation Layer — Backend"]
        E[Motion Library Lookup\nJSON clips from /motion_library/]
        F[Motion Sequencer\nclip stitching + blend]
        F2[Smooth Transition Engine\ncosine-eased cross-fade\n± hold poses between signs]
        G[Animation Stream\nJSON over REST or WebSocket]
    end

    subgraph Rendering ["Rendering Layer — Frontend"]
        H[Three.js Scene\nGLTF avatar + lighting]
        I[AvatarAnimator\nbone mapping + IK solve]
        J[Rendered 3D Avatar\nSigning in real-time]
    end

    subgraph DataPipeline ["Offline Data Pipeline"]
        V[ASL Video Dataset\nWLASL / MS-ASL] --> MP[MediaPipe Holistic\npose + hand extraction]
        MP --> N[Normalizer & Smoother\nSavitzky-Golay filter]
        N --> CL[JSON Clip Store\n/motion_library/*.json]
    end

    A1 --> B
    A2 --> B
    B --> C --> D --> E --> F --> F2 --> G --> H --> I --> J
    CL --> E

    subgraph Infra ["Infrastructure — Phase 3"]
        K[Redis Cache\ncommon sentences]
        L[S3 / Object Storage\nmotion library CDN]
        M[Docker + k8s\nscalable backend pods]
    end

    G -.-> K
    CL -.-> L
```

---

## Demo

### What the Demo Shows

A browser-based UI where:
1. User types an English sentence (e.g. *"Hello, I need help please"*).
2. Backend converts it to ASL gloss tokens and retrieves motion clips.
3. A 3D humanoid avatar signs the sequence smoothly in the viewport — **no gaps, no jumps between signs**.
4. The gloss tokens are displayed below the avatar so the user can follow along.

### Demo Setup (local)

```powershell
# Terminal 1 — Backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Demo Flow Diagram

```
User types: "I want water please"
        ↓
Backend: ["WANT", "WATER", "PLEASE"]
        ↓
Clips loaded: WANT.json → WATER.json → PLEASE.json
        ↓
Smooth transitions: cosine blend over 8 frames between each pair
        ↓
Avatar signs: [WANT] ~~ [WATER] ~~ [PLEASE]
        ↓
Gloss displayed: WANT · WATER · PLEASE
```

### Sentences to Demo

| Sentence | Expected Gloss |
|---|---|
| Hello, how are you? | HELLO HOW YOU |
| I want water please | WANT WATER PLEASE |
| My mother is sick | MOTHER SICK |
| Help me, I am lost | HELP ME LOST |
| Good morning, goodbye | GOOD MORNING GOODBYE |

---

## Running the Current Version

> **Python version**: MediaPipe requires **Python 3.10–3.12**. Python 3.13 does not expose `mp.solutions.holistic.Holistic` and will fail at extraction time.

### Prerequisites

- Python 3.10, 3.11, or 3.12 installed and on `PATH`
- Git (repo already cloned)
- WLASL videos already downloaded into `WLASL-master/start_kit/raw_videos/`
  _(see Phase 1a-1 for the download command if not done yet)_

---

### Step 1 — Set up a virtual environment

```powershell
# From the repo root
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

### Step 2 — Install data-pipeline dependencies

```powershell
pip install mediapipe opencv-python numpy scipy
# or use the requirements file:
pip install -r data_pipeline/requirements-phase1.txt
```

---

### Step 3 — Build the motion library

Run from the repo root. This processes every downloaded video through MediaPipe, normalizes and smooths the skeleton, and writes one JSON clip per gloss into `backend/motion_library/`.

```powershell
python data_pipeline/batch_process.py `
    --index WLASL-master/start_kit/WLASL_v0.3.json `
    --raw-videos WLASL-master/start_kit/raw_videos `
    --out-dir backend/motion_library `
    --max-per-gloss 5
```

Outputs:
- `backend/motion_library/*.json` — one selected clip per gloss
- `backend/motion_library/report.json` — summary of written / skipped glosses

**Dry run first** (no file writes, just validates paths):
```powershell
python data_pipeline/batch_process.py --dry-run
```

Verify the library was built:
```powershell
(Get-ChildItem backend/motion_library -Filter *.json -Exclude report.json).Count
# Expect: 50–110 files depending on how many videos downloaded cleanly
```

---

### Step 4 — Install backend dependencies

```powershell
pip install -r backend/requirements.txt
# Download the spaCy English model (required by gloss_converter.py):
python -m spacy download en_core_web_sm
```

---

### Step 5 — Start the backend

```powershell
cd backend
uvicorn main:app --reload --port 8000
```

Confirm it is up:
```powershell
curl http://localhost:8000/api/health
# Expected: {"status":"ok","motion_library_size":<N>}
```

Check which glosses are available:
```powershell
curl http://localhost:8000/api/glosses
```

The API accepts these options on `POST /api/sign`:

| Field | Default | Description |
|---|---|---|
| `text` | _(required)_ | English sentence to sign |
| `fps` | `30.0` | Playback frame rate |
| `blend_frames` | `5` | Transition frames between signs |
| `skip_unknown` | `false` | Skip words not in library (vs fingerspell) |

---

### Step 6 — Start the frontend

The frontend loads Three.js from a CDN via an `importmap` — **no npm install or build step required**. You only need a static file server (browsers block ES module imports from `file://`).

**Option A — Python built-in server (simplest):**
```powershell
cd frontend
python -m http.server 5173
# Open http://localhost:5173
```

**Option B — VS Code Live Server extension:**
Right-click `frontend/index.html` → _Open with Live Server_.

**Option C — Node `serve` (if Node is installed):**
```powershell
npx serve frontend -p 5173
```

---

### Step 7 — Use the app

1. Open `http://localhost:5173` in a browser.
2. Type a sentence in the text box (e.g. `Hello, I need help please`).
3. Click **Sign it ▶**.
4. The avatar signs the sequence; the gloss tokens appear below the viewport.
5. Use the **Speed** slider (0.25× – 3×) to slow down or speed up playback.
6. Click **Replay ↺** to replay the last animation.

---

### Step 8 — Quick API test (no frontend)

```powershell
curl -X POST http://localhost:8000/api/sign `
     -H "Content-Type: application/json" `
     -d '{"text": "hello please help"}'
```

Expected response shape:
```json
{
  "input_text": "hello please help",
  "gloss_tokens": ["HELLO", "PLEASE", "HELP"],
  "glosses_used": ["HELLO", "PLEASE", "HELP"],
  "glosses_unknown": [],
  "fps": 30.0,
  "num_frames": 145,
  "frames": [...]
}
```

---

### Troubleshooting

| Symptom | Fix |
|---|---|
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | You are on Python 3.13 — switch to 3.10–3.12 |
| `motion_library_size: 0` from `/api/health` | Run Step 3 (build motion library) first |
| `404` from `/api/sign` — "None of the gloss tokens found" | The requested words aren't in the library yet; check `report.json` for skipped glosses |
| `ModuleNotFoundError: spacy` | Run `pip install spacy` and `python -m spacy download en_core_web_sm` |
| Browser shows blank canvas | CORS or `file://` issue — use a local HTTP server (Step 6), not `file://` |
| Avatar doesn't move | Open browser DevTools console; check for a failed `fetch` to `localhost:8000` — ensure the backend is running |

---

## Phase 1 — MVP

### Goals
- Accept English text input via a simple web UI.
- Convert to ASL gloss (rule-based, no ML models needed at MVP stage).
- Retrieve pre-recorded skeleton clips and stitch together.
- Display a basic 3D avatar performing the signs in the browser.
- **Smooth transitions between signs** (no jarring instantaneous jumps).

---

### Phase 1a. Data Pipeline — Building the Sign Motion Library

#### Phase 1a-1: Source ASL video datasets — **using WLASL daily-use subset**

For MVP we use the **`--preset daily`** subset of WLASL: ~100 high-frequency everyday signs downloaded with redundancy (5 copies each) to survive corrupted files.

**Download command** (run from `WLASL-master\start_kit\`):
```powershell
python fast_video_downloader.py `
    --index WLASL_v0.3.json `
    --out raw_videos `
    --workers 8 --retries 2 `
    --nonyoutube-only --insecure `
    --preset daily --max-per-gloss 5
```

**`--preset daily` covers ~100 words across 6 categories:**

| Category | Examples |
|---|---|
| Greetings / courtesy | hello, goodbye, please, thank, sorry, yes, no |
| Question words | what, where, when, who, why, how |
| Family / people | mother, father, sister, brother, friend, baby |
| Core verbs | help, want, need, like, love, go, eat, drink, sleep, work, play |
| Feelings / state | happy, sad, angry, sick, tired, good, bad, hot, cold |
| Food / time / colours | water, food, today, tomorrow, morning, night, red, blue, green |

> **Why `--max-per-gloss 5`?** Each word gets up to 5 video instances so that even if 3–4 are corrupted or dead links you still end up with at least 1 usable video per sign.

After downloading, downloaded videos land in `WLASL-master\start_kit\raw_videos\`. For Phase 2, expand to full WLASL (21,083 videos) or add MS-ASL/ASL-LEX.

| Dataset | Description | URL |
|---|---|---|
| **WLASL** | 2000 words, 21,083 videos from Deaf signers | [WLASL on GitHub](https://github.com/dxli94/WLASL) |
| **MS-ASL** | Microsoft, 1000 classes | [MS-ASL](https://www.microsoft.com/en-us/research/project/ms-asl/) |
| **ASL-LEX** | Lexical freq. database with video | [ASL-LEX](https://asl-lex.org/) |
| **Handspeak** | Free ASL videos for common words | [handspeak.com](https://www.handspeak.com/) |
| **OpenASL** | Open-source ASL data | [OpenASL](https://github.com/chevalierNoir/OpenASL) |

---

#### Phase 1a-2: MediaPipe Skeleton Extraction

Run `mediapipe_extractor.py` on every source video to extract:
- **Pose landmarks** (33 body joints)
- **Left/Right hand landmarks** (21 keypoints each)
- **Face landmarks** (optional for lip sync, Phase 2)

**Key MediaPipe API**: `mp.solutions.holistic.Holistic`

```python
import mediapipe as mp
import cv2, json, numpy as np

def extract_clip(video_path: str, output_path: str):
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)
            frame_data = extract_landmarks(result)
            frames.append(frame_data)

    cap.release()
    clip = {"fps": fps, "frames": frames, "num_frames": len(frames)}
    with open(output_path, "w") as f:
        json.dump(clip, f)

def extract_landmarks(result) -> dict:
    def lm_to_list(lm_list):
        if lm_list is None:
            return None
        return [[lm.x, lm.y, lm.z] for lm in lm_list.landmark]

    return {
        "pose": lm_to_list(result.pose_landmarks),
        "left_hand": lm_to_list(result.left_hand_landmarks),
        "right_hand": lm_to_list(result.right_hand_landmarks),
    }
```

---

#### Phase 1a-3: Normalize and Smooth

- **Normalize**: Center all coordinates around the hip midpoint (pose landmark 23/24), scale to unit height. This makes the avatar size-independent.
- **Smooth**: Apply a Savitzky-Golay filter (window=5, poly=2) for jitter removal.

```python
from scipy.signal import savgol_filter
import numpy as np

def smooth_sequence(frames: list, window=5, poly=2) -> list:
    """Apply Savitzky-Golay filter per joint per axis."""
    arr = np.array([[f["pose"] for f in frames]])  # shape: (T, 33, 3)
    smoothed = savgol_filter(arr, window, poly, axis=0)
    return smoothed.tolist()
```

---

### Phase 1b. Skeleton Animation Storage Format

Each word/sign is stored as a single `.json` clip file:

```json
{
  "gloss": "HELLO",
  "source": "WLASL-v3",
  "fps": 30,
  "num_frames": 45,
  "joints": ["POSE_0", "POSE_1", "...", "L_HAND_0", "...", "R_HAND_0", "..."],
  "frames": [
    {
      "pose": [[x, y, z], ...],    // 33 joints
      "left_hand": [[x, y, z], ...],  // 21 joints, null if absent
      "right_hand": [[x, y, z], ...]  // 21 joints, null if absent
    }
  ]
}
```

Store these in: `motion_library/<GLOSS_TOKEN>.json`

---

### Phase 1c. Text → ASL Gloss Conversion

For MVP, use a **rule-based gloss converter**. ASL gloss follows different grammar than English (no articles, simplified verb forms). A basic approach:

1. **Tokenize** English sentence (spaCy or NLTK).
2. **Remove** articles (`a`, `an`, `the`), auxiliary verbs (`is`, `are`, `was`), and conjunctions.
3. **Lemmatize** words (e.g., `running` → `RUN`).
4. **Map** to gloss tokens (uppercase).
5. For unknown words: **fingerspell** letter-by-letter using the ASL fingerspelling library.

```python
import spacy
nlp = spacy.load("en_core_web_sm")

STOP_REMOVE = {"the", "a", "an", "is", "are", "was", "were", "be", "been"}

def text_to_gloss(sentence: str) -> list[str]:
    doc = nlp(sentence.lower())
    gloss = []
    for token in doc:
        if token.is_punct or token.text in STOP_REMOVE:
            continue
        lemma = token.lemma_.upper()
        gloss.append(lemma)
    return gloss
```

**Phase 2 upgrade**: Replace with a trained Seq2Seq transformer (e.g., mBART or T5 fine-tuned on English→ASL gloss corpora).

---

### Phase 1d. Motion Sequencer + Smooth Transitions

#### The Problem: Instantaneous Jumps Between Signs

Without blending, the avatar teleports from the last pose of one sign to the first pose of the next — this looks robotic and unnatural. The fix has two parts:

1. **Cosine-eased cross-fade** (replaces hard-cut linear blend): Uses a cosine curve `0.5 * (1 - cos(π·t))` instead of linear `t`, which decelerates into and accelerates out of the transition — matching how real signers move between signs.
2. **Hold pose before and after blend**: Append 2–3 frames of the last pose of clip A before the blend begins, so the avatar "settles" before moving to the next sign.

```python
import numpy as np, json
from pathlib import Path

LIBRARY_PATH = Path("motion_library")
BLEND_FRAMES = 8   # wider window = smoother transition
HOLD_FRAMES = 2    # brief hold at end of each sign before transitioning

def load_clip(gloss: str) -> dict | None:
    path = LIBRARY_PATH / f"{gloss}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def cosine_blend(frame_a: dict, frame_b: dict, n: int = BLEND_FRAMES) -> list[dict]:
    """Cosine-eased interpolation between two poses."""
    blended = []
    for i in range(n):
        t = i / n
        alpha = 0.5 * (1 - np.cos(np.pi * t))   # cosine ease
        frame = {}
        for key in ["pose", "left_hand", "right_hand"]:
            a = np.array(frame_a.get(key) or np.zeros((21, 3)))
            b = np.array(frame_b.get(key) or np.zeros((21, 3)))
            frame[key] = ((1 - alpha) * a + alpha * b).tolist()
        blended.append(frame)
    return blended

def sequence_clips(gloss_tokens: list[str]) -> dict:
    all_frames = []
    last_frame = None
    for token in gloss_tokens:
        clip = load_clip(token)
        if clip is None:
            clip = fingerspell_clip(token)   # fallback
        if not clip or not clip["frames"]:
            continue
        if last_frame:
            # Hold the last pose briefly so the avatar "settles"
            all_frames += [last_frame] * HOLD_FRAMES
            # Cosine-eased blend into next sign
            all_frames += cosine_blend(last_frame, clip["frames"][0])
        all_frames += clip["frames"]
        last_frame = clip["frames"][-1]
    return {"fps": 30, "num_frames": len(all_frames), "frames": all_frames}
```

> **Why cosine over linear?** Linear blending produces constant-velocity transitions that feel mechanical. Cosine easing mimics the natural acceleration/deceleration of human motion — the avatar decelerates as it reaches the hold pose and then gently accelerates into the next sign.

---

### Phase 1e. Backend Architecture

**Stack**: FastAPI (Python) + Uvicorn

```
POST /api/sign
Body: { "text": "Hello, how are you?" }
Response: { "gloss": ["HELLO", "HOW", "YOU"], "animation": { ...clip JSON... } }
```

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SignRequest(BaseModel):
    text: str

@app.post("/api/sign")
async def sign(req: SignRequest):
    gloss = text_to_gloss(req.text)
    animation = sequence_clips(gloss)
    return {"gloss": gloss, "animation": animation}
```

- For large animations, stream via **WebSocket** for better UX.
- Cache frequently requested animations in Redis (Phase 2).

---

### Phase 1f. Frontend — 3D Avatar Animation (Three.js)

**Rendering approach**: Three.js `SkeletonHelper` + bone-mapped avatar (GLB/GLTF format).

**Recommended free avatars**:
- [Ready Player Me](https://readyplayer.me/) — free GLB avatars
- [Mixamo](https://mixamo.com/) — free rigged characters

**Landmark → Bone Mapping**: Map MediaPipe joint indices to the avatar's skeleton bones. This is the core bridge step.

```javascript
// avatar_animator.js
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const POSE_BONE_MAP = {
  11: 'LeftUpperArm',
  12: 'RightUpperArm',
  13: 'LeftLowerArm',
  14: 'RightLowerArm',
  15: 'LeftHand',
  16: 'RightHand',
  23: 'LeftUpperLeg',
  24: 'RightUpperLeg',
};

class AvatarAnimator {
  constructor(scene, avatarUrl) {
    this.bones = {};
    this.frames = [];
    this.currentFrame = 0;
    this.fps = 30;
    this.loader = new GLTFLoader();
    this.loader.load(avatarUrl, (gltf) => {
      scene.add(gltf.scene);
      gltf.scene.traverse((obj) => {
        if (obj.isBone) this.bones[obj.name] = obj;
      });
    });
  }

  loadAnimation(animationData) {
    this.frames = animationData.frames;
    this.fps = animationData.fps;
    this.currentFrame = 0;
  }

  update(delta) {
    if (!this.frames.length) return;
    this.currentFrame = (this.currentFrame + delta * this.fps) % this.frames.length;
    const frame = this.frames[Math.floor(this.currentFrame)];
    this._applyPose(frame);
  }

  _applyPose(frame) {
    if (!frame.pose) return;
    for (const [idx, boneName] of Object.entries(POSE_BONE_MAP)) {
      const bone = this.bones[boneName];
      if (!bone || !frame.pose[idx]) continue;
      const [x, y, z] = frame.pose[idx];
      bone.position.set(x, y, z);
    }
  }
}
```

> **Note**: For production, use **inverse kinematics (IK)** to set bone rotations from joint positions (e.g., the `three-ik` library or custom FK/IK solver). For MVP, direct position setting gives a working prototype quickly.

---

### Phase 1g. Project Folder Structure

```
ai-sign-language-avatar/
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── gloss_converter.py       # text → ASL gloss
│   ├── motion_sequencer.py      # clip stitching + cosine blend
│   ├── motion_library/          # pre-processed JSON clips
│   │   ├── HELLO.json
│   │   ├── HOW.json
│   │   └── ...
│   ├── fingerspell/             # A-Z fingerspelling clips
│   │   ├── A.json
│   │   └── ...
│   └── requirements.txt
│
├── data_pipeline/
│   ├── mediapipe_extractor.py   # video → skeleton JSON
│   ├── normalizer.py            # normalize + smooth
│   ├── batch_process.py         # process entire dataset
│   └── scripts/
│       └── download_wlasl.sh
│
├── frontend/
│   ├── index.html
│   ├── main.js                  # Three.js scene setup
│   ├── avatar_animator.js       # pose → bone animation
│   ├── api_client.js            # calls backend /api/sign
│   ├── assets/
│   │   └── avatar.glb           # Ready Player Me avatar
│   └── style.css
│
├── tests/
│   ├── test_gloss_converter.py
│   ├── test_motion_sequencer.py
│   └── test_api.py
│
└── README.md
```

---

### Phase 1 Roadmap

| Phase | Milestone | Status |
|-------|-----------|--------|
| **1a** | Set up repo structure, install dependencies (MediaPipe, FastAPI, spaCy). | — |
| **1b** | Download WLASL daily-use subset: `--preset daily --max-per-gloss 5` (~500 videos queued). | ✅ Script ready |
| **1c** | Build `mediapipe_extractor.py`. Run on `raw_videos/` → 100 JSON clips in `motion_library/`. Build normalizer/smoother. | — |
| **1d** | Build `gloss_converter.py` (rule-based). Build `motion_sequencer.py` with **cosine-eased blend + hold frames**. Test end-to-end text → animation JSON. | — |
| **1e** | Build FastAPI backend. Test `POST /api/sign` endpoint with `curl` / pytest. | — |
| **1f** | Set up Three.js scene. Load Ready Player Me avatar. Implement landmark → bone mapping. Render looping animation from JSON. | — |
| **1g** | Connect frontend to backend. Display avatar signing from text input. Polish UI. Record demo. | — |

---

## Phase 2 — Production

### Upgrades Over MVP

| Area | Phase 1 MVP | Phase 2 |
|------|-------------|---------|
| **Gloss conversion** | Rule-based (spaCy) | Fine-tuned T5/mBART Seq2Seq model |
| **Motion library** | 100–500 words, hand-curated | 2000+ words (full WLASL vocabulary) |
| **Motion blending** | Cosine-eased interpolation | **Motion Graph** or **Neural motion matching** |
| **Avatar rendering** | Three.js with direct joint mapping | Full IK rig (Unity or Unreal Engine) |
| **Facial expression** | None | Lipsync + eyebrow/expression via BlendShapes |
| **Latency** | ~100–300ms | <50ms with streaming + caching |
| **Backend** | FastAPI synchronous | FastAPI async + Redis cache + WebSocket stream |
| **Deployment** | Local / dev server | Docker + Kubernetes + CDN |
| **Input modalities** | Text only | Text + Speech (Whisper ASR → ASL) |
| **Sign language** | ASL only | ASL + BSL + ISL (multilingual) |

---

### Phase 2 Architecture Additions

#### A. Improved Gloss Conversion (ML Model)
Train a **Seq2Seq transformer** on English→ASL gloss pairs from the [How2Sign dataset](https://how2sign.github.io/) or [Phoenix-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/).

```
English: "I want to eat pizza"
ASL Gloss: ["ME", "WANT", "EAT", "PIZZA"]
```

Use **HuggingFace** `transformers` for fine-tuning.

#### B. Motion Graph for Natural Blending
Instead of cosine interpolation, build a **motion graph** — a directed graph where nodes are key-frames of clips and edges represent valid transitions (computed by pose-space distance metric). Traversal finds the path that produces the most natural animation, eliminating velocity discontinuities at boundaries.

#### C. IK-Based Avatar in Unity
Unity's **Animation Rigging** package (`com.unity.animation.rigging`) supports full IK. Use `TwoBoneIK` constraints for arms and hands. Receive skeleton data over WebSocket and drive bones in real-time.

#### D. Voice Input (ASR → ASL)
Add OpenAI Whisper as an ASR module upstream:

```
Microphone → Whisper (STT) → English text → Gloss → Avatar
```

#### E. Deployment Architecture

```
[Client Browser / Unity WebGL]
         |  WebSocket / HTTP
[API Gateway → FastAPI + Uvicorn (k8s pods)]
         |
[Redis Cache for animation clips]
         |
[Motion Library on S3 / object storage]
```

---

### Phase 2 Dataset Recommendations

| Dataset | Words/Phrases | Notes |
|---|---|---|
| **WLASL** | 2000 words | Best starting point |
| **How2Sign** | Continuous signing | Great for sentence-level |
| **Phoenix 2014-T** | Weather reports | German, but good for Seq2Seq training |
| **ASLLVD** | 3000+ ASL signs | Produced by Rutgers/Boston |
| **ChicagoFSWild** | Fingerspelling in the wild | Great for fingerspell fallback |
| **ASL-LEX** | ~2700 signs with metadata | Academic use |

---

## Phase 3 — Avatar Enhancement (Production-Grade)

This phase focuses on making the avatar look and behave like a real signer — suitable for public-facing production deployment.

### Goals
- Photorealistic or high-quality stylized avatar with smooth, human-like motion.
- Full hand pose fidelity (21 joints per hand, finger curl, spread).
- Facial expressions and lip-sync for mouth morphemes critical to ASL grammar.
- Sub-50ms end-to-end latency with WebSocket streaming.
- Accessibility compliance (WCAG 2.1 AA for the surrounding UI).

---

### Phase 3a. Full IK Rig & Bone-Level Animation

Replace direct joint position setting with a proper **FK/IK pipeline**:

| Component | Tool |
|---|---|
| Arm IK | `three-ik` library or custom Two-Bone IK |
| Finger pose | Direct FK from hand landmark angles |
| Wrist twist | Rodrigues rotation from palm normal vector |
| Shoulder raise | Heuristic from elbow height vs shoulder height |

```javascript
// Compute bone rotation from two landmark positions (FK approach)
function computeBoneRotation(parentPos, childPos, bindQuaternion) {
  const dir = new THREE.Vector3().subVectors(childPos, parentPos).normalize();
  const q = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),   // bind-pose direction
    dir
  );
  return q.multiply(bindQuaternion.clone().invert()).multiply(bindQuaternion);
}
```

---

### Phase 3b. High-Fidelity Hand Rendering

ASL is hand-dominant — hand quality is critical for linguistic accuracy.

- Use **all 21 MediaPipe hand landmarks** (not just wrist proxy).
- Compute **finger joint angles** from landmark triplets and apply as FK rotations.
- Add a **hand confidence filter**: if MediaPipe confidence < 0.7, blend to the previous valid frame rather than showing a collapsed hand.

```python
# In data_pipeline/normalizer.py — confidence-aware smoothing
def filter_low_confidence_frames(frames, threshold=0.7):
    last_valid = None
    for frame in frames:
        conf = frame.get("hand_confidence", 1.0)
        if conf < threshold and last_valid:
            frame["left_hand"] = last_valid["left_hand"]
            frame["right_hand"] = last_valid["right_hand"]
        else:
            last_valid = frame
    return frames
```

---

### Phase 3c. Facial Expression & Mouth Morphemes

In ASL, **facial expressions carry grammatical meaning** (e.g., raised eyebrows signal a yes/no question; furrowed brows signal a WH-question). Ignoring them makes the avatar linguistically incorrect.

**Implementation options (in order of effort):**

| Approach | Quality | Effort |
|---|---|---|
| Pre-defined expression presets per gloss type | Low | Low |
| MediaPipe FaceMesh (468 landmarks) mapped to BlendShapes | Medium | Medium |
| **Recommended**: ARKit-compatible BlendShape set on GLTF avatar | High | High |

```python
# Phase 3 extractor addition — face landmarks
"face": lm_to_list(result.face_landmarks)  # 468 points
```

Map key face landmarks to GLTF morph targets:
- `browInnerUp`, `browOuterUpLeft/Right` → question type
- `jawOpen` → mouth morpheme
- `cheekPuff` → classifier handshape cues

---

### Phase 3d. Smooth Transition Deep-Dive (Production Fix)

Beyond Phase 1's cosine blend, production requires:

1. **Velocity continuity**: Match the exit velocity of clip A to the entry velocity of clip B — prevents the "rubber band snap" even with cosine easing.
2. **Motion Graph traversal**: Pre-compute a graph of pose-space distances between clip endpoints. At runtime, pick the transition path with lowest discontinuity cost.
3. **Spring-damper blending**: Model the transition as an under-damped spring — natural overshoot then settle, like real human motion.

```python
def spring_blend(pos_a, vel_a, pos_b, n_frames, stiffness=80.0, damping=12.0):
    """Spring-damper simulation for physically-based transitions."""
    dt = 1.0 / 30.0
    pos, vel = np.array(pos_a), np.array(vel_a)
    frames = []
    for _ in range(n_frames):
        acc = stiffness * (np.array(pos_b) - pos) - damping * vel
        vel = vel + acc * dt
        pos = pos + vel * dt
        frames.append(pos.tolist())
    return frames
```

4. **Frontend-side interpolation**: Even if the backend sends frames at 30 fps, the browser renders at 60–120 Hz. Use `THREE.AnimationMixer` or manual lerp between consecutive frames at sub-frame precision in `requestAnimationFrame`.

---

### Phase 3e. Performance & Scalability

| Concern | Solution |
|---|---|
| Large JSON animation payloads (~500 KB per sentence) | Quantize float32 → int16 (2× compression); stream over WebSocket frame-by-frame |
| Cold-start latency on first request | Pre-warm cache for top-500 common sentences on startup |
| Concurrent users | Stateless FastAPI pods behind load balancer; animation JSON is read-only |
| Motion library size (2000+ clips × ~100 KB each = ~200 MB) | Serve from CDN with aggressive cache headers; lazy-load on first use |
| Avatar mesh download (~5–15 MB GLB) | Compress with Draco mesh compression; cache in browser IndexedDB |

---

### Phase 3f. Accessibility & Compliance

| Requirement | Implementation |
|---|---|
| WCAG 2.1 AA colour contrast | Design system with 4.5:1 minimum ratio |
| Keyboard navigation | All controls accessible without mouse |
| Screen reader support | ARIA labels on all interactive elements |
| Pause / replay controls | Allow user to replay any sign at half speed |
| Caption sync | Display current gloss token highlighted as avatar signs it |
| Speed control | 0.5× / 1× / 1.5× playback speed slider |

---

### Phase 3g. Observability & Quality Monitoring

- **Landmark quality score** per clip (% frames with full hand visibility) stored in clip metadata.
- **User feedback loop**: thumbs up/down on each signed word → feeds back into clip ranking to prefer the clearest instance.
- **A/B testing framework**: serve different blend window sizes to different users; measure preference via feedback.
- **Logging**: structured JSON logs per request (gloss tokens, missing clips, fallback rate, latency).

---

### Phase 3 Roadmap

| Phase | Milestone |
|-------|-----------|
| **3a** | Replace position-based bone driving with FK/IK solver. Validate wrist and elbow angles match source video. |
| **3b** | Implement full 21-joint finger FK. Add confidence-based hand frame filtering in data pipeline. |
| **3c** | Add MediaPipe FaceMesh extraction. Map to BlendShape targets on avatar. Test question-type expressions. |
| **3d** | Implement spring-damper transition blending. Pre-compute motion graph for top-200 sign pairs. |
| **3e** | Quantize animation JSON. Add WebSocket streaming. Deploy Redis cache. Set up CDN for motion library. |
| **3f** | Accessibility audit. Add caption sync overlay. Add speed control. |
| **3g** | Add structured logging, landmark quality scores, and user feedback API. |

---

## Key Dependencies

```
# backend/requirements.txt
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
mediapipe>=0.10.14
opencv-python>=4.9.0
spacy>=3.7.4
scipy>=1.13.0
numpy>=1.26.4
redis>=5.0.4        # Phase 2+
websockets>=12.0    # Phase 2+

# NLP model
# python -m spacy download en_core_web_sm
```

```json
// frontend/package.json (Vite + Three.js)
{
  "dependencies": {
    "three": "^0.164.1",
    "vite": "^5.2.0"
  }
}
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Landmark → bone mapping is inaccurate | Use FK/IK solver; test with multiple avatar rigs |
| MediaPipe fails on occluded hands | Filter low-confidence frames; fill gaps with last valid frame |
| Gloss converter misses words | Fingerspell fallback for any unknown token |
| High latency on clip stitching | Pre-cache top-1000 common sentences; stream clips progressively |
| **Choppy / instantaneous transitions between signs** | **Phase 1: cosine-eased blend + hold frames (8-frame window). Phase 3: spring-damper + motion graph** |
| Avatar looks robotic | Phase 3 IK rig + hand FK + facial expressions |
| Dataset licensing restrictions | WLASL is research-use; build custom clips for commercial use |
| Performance degradation at scale | Stateless pods + Redis cache + CDN-served motion library |

---

## Testing Guide

A layer-by-layer approach: verify each component independently before wiring everything together.

---

### Layer 0 — Verify Downloaded Videos

```powershell
# Count how many videos actually downloaded
(Get-ChildItem WLASL-master\start_kit\raw_videos -File).Count

# Spot-check a file is not zero bytes (corrupt)
Get-ChildItem WLASL-master\start_kit\raw_videos -File |
    Where-Object { $_.Length -lt 1000 } |
    Select-Object Name, Length
# Any files listed here are corrupt/incomplete — delete and re-run the downloader
```

Play a few videos manually to confirm they show a signer:
```powershell
# Open a random video in Windows Media Player
$vids = Get-ChildItem WLASL-master\start_kit\raw_videos -Filter *.mp4
Start-Process $vids[0].FullName
```

---

### Layer 1 — Test MediaPipe Skeleton Extraction

Run the extractor on a **single video** first:

```python
# data_pipeline/mediapipe_extractor.py
python data_pipeline/mediapipe_extractor.py \
    --video WLASL-master/start_kit/raw_videos/<some_video_id>.mp4 \
    --out backend/motion_library/TEST.json
```

Inspect the output JSON to confirm it has frames with pose/hand data:

```python
import json
clip = json.load(open("backend/motion_library/TEST.json"))
print(f"fps={clip['fps']}  frames={clip['num_frames']}")
print("First frame keys:", clip['frames'][0].keys())
print("Pose landmarks in frame 0:", len(clip['frames'][0]['pose'] or []))
```

Expected output:
```
fps=25.0  frames=48
First frame keys: dict_keys(['pose', 'left_hand', 'right_hand'])
Pose landmarks in frame 0: 33
```

---

### Layer 2 — Test Gloss Converter

```python
# From repo root
python -c "
from backend.gloss_converter import text_to_gloss
tests = [
    'Hello, how are you?',
    'I want water please',
    'My mother is sick',
    'What is your name?',
]
for t in tests:
    print(f'  {t!r}')
    print(f'  → {text_to_gloss(t)}')
    print()
"
```

Expected: articles/auxiliaries removed, lemmas uppercased, e.g.:
```
'Hello, how are you?' → ['HELLO', 'HOW', 'YOU']
'I want water please' → ['WANT', 'WATER', 'PLEASE']
```

---

### Layer 3 — Test Motion Sequencer

```python
python -c "
from backend.motion_sequencer import sequence_clips
result = sequence_clips(['HELLO', 'WATER', 'HELP'])
print(f'Total frames: {result[\"num_frames\"]}')
print(f'FPS: {result[\"fps\"]}')
print(f'First frame keys: {list(result[\"frames\"][0].keys())}')
"
```

**Test smooth transition specifically:**
```python
python -c "
from backend.motion_sequencer import sequence_clips
import json

result = sequence_clips(['HELLO', 'WATER'])
frames = result['frames']
# Check no hard jumps — adjacent frames should have low max joint delta
import numpy as np
for i in range(1, len(frames)):
    a = np.array(frames[i-1]['pose'] or np.zeros((33,3)))
    b = np.array(frames[i  ]['pose'] or np.zeros((33,3)))
    delta = np.max(np.linalg.norm(b - a, axis=1))
    if delta > 0.15:
        print(f'Frame {i}: large jump detected ({delta:.3f}) — check blend window')
print('Transition check complete')
"
```

If a gloss JSON clip is missing, it should fall back to fingerspelling without crashing.

---

### Layer 4 — Test the FastAPI Backend

**Start the server:**
```powershell
cd backend
uvicorn main:app --reload --port 8000
```

**Test with curl:**
```powershell
# Basic sign request
curl -X POST http://localhost:8000/api/sign `
     -H "Content-Type: application/json" `
     -d '{"text": "hello please help"}'

# Expected response shape:
# { "gloss": ["HELLO", "PLEASE", "HELP"],
#   "animation": { "fps": 30, "num_frames": ..., "frames": [...] } }
```

**Test with pytest** (add to `tests/test_api.py`):
```python
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_sign_endpoint_returns_gloss():
    r = client.post("/api/sign", json={"text": "hello water"})
    assert r.status_code == 200
    body = r.json()
    assert "gloss" in body
    assert "HELLO" in body["gloss"]
    assert body["animation"]["num_frames"] > 0
```

Run tests:
```powershell
pytest tests/ -v
```

---

### Layer 5 — Test the Frontend (Three.js)

1. Start the backend (Layer 4 above).
2. In a second terminal, start the Vite dev server:
   ```powershell
   cd frontend
   npm install
   npm run dev
   # Opens at http://localhost:5173
   ```
3. Open `http://localhost:5173` in a browser.
4. Type `hello water please` in the text box and click **Sign**.
5. Verify the 3D avatar animates — arms/hands should move through the sequence **without visible jumps between words**.

**Quick browser console test** (no avatar needed yet — just check the API response):
```javascript
// Paste in DevTools console while frontend is open
fetch('http://localhost:8000/api/sign', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'hello help water'})
}).then(r => r.json()).then(d => {
  console.log('Gloss:', d.gloss);
  console.log('Frame count:', d.animation.num_frames);
});
```

---

### End-to-End Smoke Test Checklist

| # | Check | Pass when... |
|---|-------|-------------|
| 1 | Videos downloaded | `raw_videos/` has ≥50 `.mp4` files, none under 1 KB |
| 2 | Extractor runs | `motion_library/HELLO.json` exists with `num_frames > 10` |
| 3 | Gloss converter | `text_to_gloss("hello how are you")` → `["HELLO", "HOW", "YOU"]` |
| 4 | Sequencer | `sequence_clips(["HELLO","WATER"])` returns dict with frames list |
| 5 | Smooth transitions | No frame-to-frame joint delta > 0.15 at sign boundaries |
| 6 | API responds | `POST /api/sign` returns 200 with gloss + animation JSON |
| 7 | Frontend renders | Avatar moves arms **smoothly** when a sentence is submitted |
