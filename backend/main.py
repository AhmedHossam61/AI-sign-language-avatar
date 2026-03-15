"""FastAPI backend for the AI Sign Language Avatar.

Endpoints:
    POST /api/sign          → full animation JSON from English text
    GET  /api/glosses       → list of known gloss tokens in the motion library
    GET  /api/health        → liveness check

Run with:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Allow importing sibling modules (gloss_converter, motion_sequencer).
sys.path.insert(0, str(Path(__file__).parent))

from gloss_converter import text_to_gloss
from motion_sequencer import MOTION_LIBRARY_DIR, sequence_glosses, sequence_text

app = FastAPI(
    title="AI Sign Language Avatar API",
    version="1.0.0",
    description="Converts English text to ASL skeleton animation JSON.",
)

# Allow the Three.js frontend (any origin during development) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SignRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, example="Hello, how are you today?")
    fps: float = Field(default=30.0, ge=1.0, le=120.0)
    blend_frames: int = Field(default=5, ge=0, le=30)
    skip_unknown: bool = Field(default=False)


class SignResponse(BaseModel):
    input_text: str
    gloss_tokens: list[str]
    glosses_used: list[str]
    glosses_unknown: list[str]
    fps: float
    num_frames: int
    frames: list[dict[str, Any]]


class GlossListResponse(BaseModel):
    glosses: list[str]
    count: int


class HealthResponse(BaseModel):
    status: str
    motion_library_size: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, tags=["utility"])
def health() -> HealthResponse:
    """Liveness / readiness check."""
    clips = list(MOTION_LIBRARY_DIR.glob("*.json"))
    # Exclude report.json and other non-clip files.
    clips = [c for c in clips if c.stem.isupper() or c.stem[0].isupper()]
    return HealthResponse(status="ok", motion_library_size=len(clips))


@app.get("/api/glosses", response_model=GlossListResponse, tags=["utility"])
def list_glosses() -> GlossListResponse:
    """Return all available gloss tokens in the motion library."""
    clips = sorted(
        p.stem for p in MOTION_LIBRARY_DIR.glob("*.json")
        if p.stem != "report" and p.stem.replace("_", "").isalpha()
    )
    return GlossListResponse(glosses=clips, count=len(clips))


@app.post("/api/sign", response_model=SignResponse, tags=["animation"])
def sign(req: SignRequest) -> SignResponse:
    """Convert English text to an ASL skeleton animation.

    Returns per-frame pose and hand landmark data ready for the Three.js
    avatar animator.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be blank")

    try:
        animation = sequence_text(
            req.text,
            fps=req.fps,
            blend_frames=req.blend_frames,
            skip_unknown=req.skip_unknown,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if animation["num_frames"] == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                "No animation frames could be produced. "
                "None of the gloss tokens were found in the motion library."
            ),
        )

    return SignResponse(**animation)
