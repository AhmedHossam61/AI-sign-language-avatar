# AI Sign Language Avatar

A real-time system that converts spoken English (via ASR) into ASL skeleton animations played by a 3D avatar in the browser.

> Full multi-phase implementation plan (architecture, roadmap, testing guide) → [`plan.md`](plan.md)

---

## Current Architecture (Phase 1 — Implemented)

What is running right now, built from the actual source files:

```mermaid
flowchart TD
    subgraph Offline ["Offline Data Pipeline (run once)"]
        V["WLASL Videos\n(raw_videos/*.mp4)"]
        MP["mediapipe_extractor.py\nMediaPipe Holistic\npose 33pt + hands 21pt each"]
        NM["normalizer.py\nhip-center normalize\nSavitzky-Golay smooth"]
        CL["motion_library/*.json\none clip per gloss"]
        V --> MP --> NM --> CL
    end

    subgraph Input ["User Input"]
        MIC["Microphone\n(live audio)"]
        ASR["ASR Engine\nWhisper / streaming STT\nspeech → text transcript"]
        MIC -->|"audio stream"| ASR
    end

    subgraph Backend ["Backend — FastAPI + Uvicorn (localhost:8000)"]
        GC["gloss_converter.py\nrule-based tokenizer\nspaCy optional, regex fallback\nstop-word removal + lemmatize"]
        MS["motion_sequencer.py\nclip lookup + stitch\nlinear blend between signs\nfingerspell fallback"]
        API["main.py — REST API\nGET  /api/health\nGET  /api/glosses\nPOST /api/sign"]
        GC --> MS --> API
    end

    subgraph Frontend ["Frontend — Three.js r165 via CDN (localhost:5173)"]
        HC["index.html\nimportmap → CDN\nno build step"]
        AC["api_client.js\nfetch POST /api/sign"]
        AA["avatar_animator.js\nbone mapping\nupdate loop"]
        UI["Browser\n3D avatar + speed slider\ngloss display + replay"]
        HC --> AC --> AA --> UI
    end

    CL -->|"loaded at startup"| MS
    API -->|"animation JSON\n{fps, num_frames, frames[]}"| AC
    ASR -->|"English transcript"| AC

    style MIC fill:#5bc0de,color:#000
    style ASR fill:#5bc0de,color:#000
```

### What each file does today

| File | Role | Notes |
|---|---|---|
| `data_pipeline/mediapipe_extractor.py` | Video → skeleton JSON | Pose (33 joints) + left/right hand (21 joints each) |
| `data_pipeline/normalizer.py` | Normalize + smooth | Hip-center origin, unit height, Savitzky-Golay filter |
| `data_pipeline/batch_process.py` | Batch driver | Processes all raw videos → `motion_library/` |
| `backend/gloss_converter.py` | English → ASL gloss | Rule-based; uses spaCy if installed, regex fallback otherwise |
| `backend/motion_sequencer.py` | Clip stitching | Linear blend transitions; fingerspell fallback for unknown tokens |
| `backend/main.py` | FastAPI server | `/api/health`, `/api/glosses`, `/api/sign` |
| `frontend/avatar_animator.js` | Bone animation | MediaPipe landmark index → Three.js bone position |
| `frontend/api_client.js` | API bridge | `fetch` to `localhost:8000/api/sign` |
| `frontend/main.js` | Three.js scene | Camera, lighting, render loop, speed control |
| `frontend/index.html` | Entry point | importmap loads Three.js r165 from CDN — no npm needed |

### What is NOT yet implemented (planned in later phases)

| Feature | Planned phase |
|---|---|
| Cosine / spring-damper transition blending | Phase 1d / Phase 3d |
| Fingerspell clips (A–Z) | Phase 1 |
| Voice input (Whisper ASR) | Phase 3 |
| Redis caching | Phase 2 |
| WebSocket streaming | Phase 2 |
| Full FK/IK bone solver | Phase 3a |
| Facial expressions / BlendShapes | Phase 3c |
| Docker + Kubernetes deployment | Phase 2 |

---

## Quick Start

> **Python version**: MediaPipe requires **Python 3.10–3.12**. Python 3.13 does not expose `mp.solutions.holistic.Holistic`.

### 1 — Set up environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2 — Install data-pipeline dependencies

```powershell
pip install -r data_pipeline/requirements-phase1.txt
```

### 3 — Build the motion library

Run from the repo root after downloading WLASL videos into `WLASL-master/start_kit/raw_videos/`:

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

**Dry run first** (no file writes):
```powershell
python data_pipeline/batch_process.py --dry-run
```

### 4 — Install backend dependencies

```powershell
pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm
```

### 5 — Start the backend

```powershell
cd backend
uvicorn main:app --reload --port 8000
```

Verify:
```powershell
curl http://localhost:8000/api/health
# {"status":"ok","motion_library_size":<N>}
```

### 6 — Start the frontend

No npm or build step — Three.js loads from CDN via importmap. You only need any static file server:

```powershell
# Option A — Python (simplest)
cd frontend
python -m http.server 5173

# Option B — VS Code Live Server
# Right-click frontend/index.html → Open with Live Server

# Option C — Node
npx serve frontend -p 5173
```

Open `http://localhost:5173`, type a sentence, click **Sign it ▶**.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | Switch to Python 3.10–3.12 |
| `motion_library_size: 0` from `/api/health` | Run Step 3 first |
| `404` from `/api/sign` — "None of the gloss tokens found" | Words not in library — check `report.json` for skipped glosses |
| `ModuleNotFoundError: spacy` | `pip install spacy && python -m spacy download en_core_web_sm` |
| Browser shows blank canvas | Must serve via HTTP, not `file://` — use Step 6 |
| Avatar doesn't move | Check browser DevTools console for a failed fetch to `localhost:8000` |

---

## Suggested Architectures

The following architectures address two core technical challenges identified during development:

1. **Real-time processing** — the current system waits for a complete sentence before translating, causing latency during live speech.
2. **Arabic sign language data scarcity** — limited datasets, plus dialect variation (Egyptian, Gulf/Saudi, Sudanese), make a unified model harder to build.

---

### Challenge 1 — Real-time Speech-to-Sign (Two Solutions)

#### Solution A — LLM-Assisted Segmentation

An LLM layer sits between the ASR output and the Gloss converter. It reads the streaming token stream, decides when a meaningful phrase is complete, and forwards only that chunk for sign translation — preserving cross-phrase context.

**Trade-offs:** higher API cost, extra latency per LLM call, accuracy depends on model context quality.

```mermaid
flowchart TD
    MIC["Microphone\n(live audio)"]
    ASR["ASR Engine\nWhisper / streaming STT\npartial transcripts"]
    LLM["LLM Segmenter\nGPT-4o / Claude\ndetects phrase boundary\nmaintains context window"]
    GC["gloss_converter.py\nEnglish phrase → ASL Gloss\nrule-based + spaCy"]
    MS["motion_sequencer.py\nclip lookup + blend"]
    API["FastAPI\nPOST /api/sign"]
    AA["avatar_animator.js\nThree.js bone animation"]
    UI["Browser\n3D Avatar — real-time"]

    MIC -->|"audio stream"| ASR
    ASR -->|"partial text tokens"| LLM
    LLM -->|"complete phrase\n+ context"| GC
    GC --> MS --> API
    API -->|"animation JSON\nover WebSocket"| AA --> UI

    style LLM fill:#f0ad4e,color:#000
    style ASR fill:#5bc0de,color:#000
```

#### Solution B — Streaming Segmentation (Silence / Punctuation Detection)

Instead of an LLM, the stream is split into chunks using lightweight heuristics already available in the ASR output. Each chunk is sent directly to the Gloss converter.

**Segmentation triggers:**
- Speaker pauses (silence gap ≥ threshold, e.g. 400 ms)
- Punctuation characters (`. , ? !`) appear in the partial transcript
- Transcript text is stable (no token change) for a short window (e.g. 300 ms)

**Trade-offs:** lower cost, simpler implementation, but real-world accuracy depends on segmentation quality — best validated through testing.

```mermaid
flowchart TD
    MIC["Microphone\n(live audio)"]
    ASR["ASR Engine\nWhisper / streaming STT\npartial transcripts"]

    subgraph SEG ["Streaming Segmenter (lightweight)"]
        SD["Silence Detector\ngap ≥ 400 ms → flush"]
        PD["Punctuation Detector\n. , ? ! → flush"]
        ST["Stability Watcher\nno change for 300 ms → flush"]
        SD & PD & ST -->|"segment ready"| FLUSH["Flush chunk"]
    end

    GC["gloss_converter.py\nEnglish chunk → ASL Gloss"]
    MS["motion_sequencer.py\nclip lookup + blend"]
    API["FastAPI\nPOST /api/sign"]
    AA["avatar_animator.js\nThree.js bone animation"]
    UI["Browser\n3D Avatar — real-time"]

    MIC -->|"audio stream"| ASR
    ASR -->|"partial tokens"| SEG
    FLUSH -->|"text chunk"| GC
    GC --> MS --> API
    API -->|"animation JSON\nover WebSocket"| AA --> UI

    style SEG fill:#5cb85c,color:#000
    style ASR fill:#5bc0de,color:#000
```

---

### Comparison — Solution A vs Solution B

| Dimension | Solution A — LLM Segmenter | Solution B — Streaming Segmentation |
|---|---|---|
| Segmentation accuracy | High (context-aware) | Medium (heuristic-based) |
| Implementation complexity | Medium | Low |
| Latency | Higher (LLM round-trip) | Lower (local heuristics) |
| Cost | Higher (LLM API calls) | Near zero |
| Arabic dialect handling | Strong (LLM generalises) | Depends on ASR punctuation |
| Best for | Production / accuracy-first | MVP / speed-first |

---

### Challenge 2 — Arabic Sign Language Data Scarcity

Arabic Sign Language (ArSL) has no large public dataset, and signs vary by dialect:

| Dialect | Region |
|---|---|
| Egyptian ArSL | Egypt |
| Gulf / Saudi ArSL | KSA, UAE, Kuwait, Qatar |
| Sudanese ArSL | Sudan |

The architecture below shows how the existing pipeline would extend to support ArSL, with a dialect router and separate motion libraries per dialect.

```mermaid
flowchart TD
    INPUT["Arabic Text Input\n(from ASR or typed)"]
    DETECT["Dialect Detector\nclassifier or user-selected"]

    subgraph LIBS ["Per-Dialect Motion Libraries"]
        EG["Egyptian ArSL\nmotion_library_eg/"]
        GULF["Gulf ArSL\nmotion_library_gulf/"]
        SU["Sudanese ArSL\nmotion_library_su/"]
    end

    GC["ArSL Gloss Converter\nArabic NLP\ncameltools / farasa"]
    MS["motion_sequencer.py\nclip lookup + blend\n+ fingerspell fallback"]
    API["FastAPI /api/sign"]
    AA["avatar_animator.js"]
    UI["Browser — 3D Avatar"]

    INPUT --> DETECT
    DETECT -->|"dialect tag"| GC
    GC -->|"loads correct lib"| LIBS
    EG & GULF & SU --> MS
    MS --> API --> AA --> UI

    style DETECT fill:#9b59b6,color:#fff
    style LIBS fill:#ecf0f1,color:#000
```

---

### Future Alternative — End-to-End Vision-Language Model

> **Status: Research stage — estimated 6–12 months of development.**

A generative model that produces sign animations directly from text, with **no pre-built motion library**. Requires a large dataset and a custom-trained Vision-Language Model (VLM).

**Requirements:**
- Large ArSL video dataset (currently unavailable publicly)
- Custom VLM fine-tuned on sign generation
- Significant compute for training

```mermaid
flowchart TD
    TEXT["Input Text\n(Arabic / English)"]
    VLM["Vision-Language Model\nfine-tuned for sign generation\ntransforms text → pose sequence\nno motion library needed"]
    POSE["Generated Pose Sequence\nskeleton keyframes"]
    AA["avatar_animator.js\nThree.js bone animation"]
    UI["Browser — 3D Avatar"]

    DATA["Training Data\nArSL videos + transcripts\n(to be collected)"]
    TRAIN["Model Training\ncustom VLM\n~6–12 months"]

    DATA --> TRAIN --> VLM
    TEXT --> VLM --> POSE --> AA --> UI

    style VLM fill:#e74c3c,color:#fff
    style TRAIN fill:#c0392b,color:#fff
    style DATA fill:#e67e22,color:#fff
```
