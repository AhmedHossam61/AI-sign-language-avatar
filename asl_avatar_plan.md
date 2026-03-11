# AI Sign Language Avatar — Technical Implementation Plan

## Overview

A real-time system that converts English text into ASL (American Sign Language) sign animations played by a 3D avatar. The system targets **low latency** by avoiding generative/diffusion models in favor of a **skeleton-based animation library** built from real signer recordings.

---

## System Architecture

```mermaid
flowchart TD
    A[User Input: English Text] --> B[Text Preprocessor]
    B --> C[ASL Gloss Converter\nnlpconnect / custom rule-based]
    C --> D[Gloss Tokenizer]
    D --> E[Motion Library Lookup\nJSON animation clips]
    E --> F[Motion Sequencer\nblend + stitch clips]
    F --> G[Skeleton Animation Stream\nJSON / WebSocket]
    G --> H[Frontend Avatar\nThree.js + SkeletonHelper]
    H --> I[Rendered 3D Avatar\nSigning in real-time]

    subgraph Data Pipeline [Offline Data Pipeline]
        V[ASL Video Dataset] --> MP[MediaPipe Holistic\npose+hand extraction]
        MP --> N[Normalizer & Smoother]
        N --> J[JSON Clip Store\n/motion_library/*.json]
    end
```

---

## Phase 1 — MVP

### Goals
- Accept English text input via a simple web UI.
- Convert to ASL gloss (rule-based, no ML models needed at MVP stage).
- Retrieve pre-recorded skeleton clips and stitch together.
- Display a basic 3D avatar performing the signs in the browser.

---

### 1. Data Pipeline — Building the Sign Motion Library

#### Step 1: Source ASL video datasets

| Dataset | Description | URL |
|---|---|---|
| **WLASL** | 2000 words, 21,083 videos from Deaf signers | [WLASL on GitHub](https://github.com/dxli94/WLASL) |
| **MS-ASL** | Microsoft, 1000 classes | [MS-ASL](https://www.microsoft.com/en-us/research/project/ms-asl/) |
| **ASL-LEX** | Lexical freq. database with video | [ASL-LEX](https://asl-lex.org/) |
| **Handspeak** | Free ASL videos for common words | [handspeak.com](https://www.handspeak.com/) |
| **OpenASL** | Open-source ASL data | [OpenASL](https://github.com/chevalierNoir/OpenASL) |

Start with **WLASL** — it has the largest vocabulary and is widely used in research. For MVP, hand-curate ~200–500 high-frequency English words.

---

#### Step 2: MediaPipe Skeleton Extraction

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

#### Step 3: Normalize and Smooth

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

### 2. Skeleton Animation Storage Format

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

### 3. Text → ASL Gloss Conversion

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

### 4. Motion Sequencer

Concatenates clips with a **linear blend** transition to avoid hard cuts:

```python
import numpy as np, json
from pathlib import Path

LIBRARY_PATH = Path("motion_library")
BLEND_FRAMES = 5

def load_clip(gloss: str) -> dict | None:
    path = LIBRARY_PATH / f"{gloss}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def blend_frames(clip_a_last, clip_b_first, n=BLEND_FRAMES):
    """Linear interpolation between end of clip A and start of clip B."""
    blended = []
    for i in range(n):
        alpha = i / n
        frame = {}
        for key in ["pose", "left_hand", "right_hand"]:
            a = np.array(clip_a_last[key] or np.zeros((21,3)))
            b = np.array(clip_b_first[key] or np.zeros((21,3)))
            frame[key] = ((1 - alpha) * a + alpha * b).tolist()
        blended.append(frame)
    return blended

def sequence_clips(gloss_tokens: list[str]) -> dict:
    all_frames = []
    last_frame = None
    for token in gloss_tokens:
        clip = load_clip(token)
        if clip is None:
            clip = fingerspell_clip(token)  # fallback
        if last_frame and clip["frames"]:
            all_frames += blend_frames(last_frame, clip["frames"][0])
        all_frames += clip["frames"]
        last_frame = clip["frames"][-1] if clip["frames"] else last_frame
    return {"fps": 30, "num_frames": len(all_frames), "frames": all_frames}
```

---

### 5. Backend Architecture

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

### 6. Frontend — 3D Avatar Animation (Three.js)

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
      // Compute and apply rotation via IK or direct joint angles
      bone.position.set(x, y, z);
    }
  }
}
```

> **Note**: For production, use **inverse kinematics (IK)** to set bone rotations from joint positions (e.g., the `three-ik` library or custom FK/IK solver). For MVP, direct position setting gives a working prototype quickly.

---

### 7. Project Folder Structure

```
ai-sign-language-avatar/
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── gloss_converter.py       # text → ASL gloss
│   ├── motion_sequencer.py      # clip stitching
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

### 8. MVP Step-by-Step Roadmap

| Week | Milestone |
|------|-----------|
| **1** | Set up repo structure, install dependencies (MediaPipe, FastAPI, spaCy). Download WLASL dataset (or subset of 100 common words). |
| **2** | Build `mediapipe_extractor.py`. Process 100 words → 100 JSON clips. Build normalizer/smoother. |
| **3** | Build `gloss_converter.py` (rule-based). Build `motion_sequencer.py` with linear blend. Test end-to-end text → animation JSON. |
| **4** | Build FastAPI backend. Test `POST /api/sign` endpoint. |
| **5** | Set up Three.js scene. Load Ready Player Me avatar. Implement landmark → bone mapping. Render basic looping animation from JSON. |
| **6** | Connect frontend to backend. Display avatar signing from text input. Polish UI. Record demo. |

**Total estimated MVP time: 6 weeks (1 developer)**

---

## Phase 2 — Production

### Upgrades Over MVP

| Area | MVP | Phase 2 |
|------|-----|---------|
| **Gloss conversion** | Rule-based (spaCy) | Fine-tuned T5/mBART Seq2Seq model |
| **Motion library** | 100–500 words, hand-curated | 2000+ words (full WLASL vocabulary) |
| **Motion blending** | Linear interpolation | **Motion Graph** or **Neural motion matching** |
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
Instead of simple linear interpolation, build a **motion graph** — a directed graph where nodes are key-frames of clips and edges represent valid transitions (computed by distance metric on pose similarity). Traversal finds the path that produces the most natural animation.

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
redis>=5.0.4        # Phase 2
websockets>=12.0    # Phase 2

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
| MediaPipe fails on occluded hands | Filter low-confidence frames; fill gaps with interpolation |
| Gloss converter misses words | Fingerspell fallback for any unknown token |
| High latency on clip stitching | Pre-cache top-1000 common sentences; stream clips progressively |
| Choppy transitions between signs | Increase blend window; use motion graph in Phase 2 |
| Dataset licensing restrictions | WLASL is research-use; build custom clips for commercial use |
