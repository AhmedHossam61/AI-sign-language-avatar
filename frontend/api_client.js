/**
 * api_client.js — talks to the FastAPI backend.
 */

const API_BASE = "http://localhost:8000";

/**
 * Send English text to the backend and receive an animation payload.
 * @param {string} text
 * @param {object} opts
 * @returns {Promise<object>} animation JSON
 */
export async function fetchAnimation(text, opts = {}) {
  const body = {
    text,
    fps: opts.fps ?? 30,
    blend_frames: opts.blendFrames ?? 5,
    skip_unknown: opts.skipUnknown ?? false,
  };

  const res = await fetch(`${API_BASE}/api/sign`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Unknown error from backend");
  }

  return res.json();
}
