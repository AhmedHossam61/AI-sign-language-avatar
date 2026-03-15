/**
 * main.js — Babylon.js scene setup + UI wiring.
 *
 * Ties together:
 *   - Babylon.js Engine / Scene / Camera / Lighting  (replaces Three.js)
 *   - AvatarAnimator (avatar_animator.js)
 *   - fetchAnimation (api_client.js)
 *
 * BABYLON is a global loaded from the CDN <script> tag in index.html.
 */

import { AvatarAnimator } from "./avatar_animator.js";
import { fetchAnimation }  from "./api_client.js";

// ─── Canvas + Engine ─────────────────────────────────────────────────────────
const canvas = document.getElementById("render-canvas");

const engine = new BABYLON.Engine(canvas, /* antialias */ true, {
  preserveDrawingBuffer: true,
  stencil: true,
});

// ─── Scene ───────────────────────────────────────────────────────────────────
const scene = new BABYLON.Scene(engine);
scene.clearColor = new BABYLON.Color4(0.102, 0.114, 0.153, 1); // #1a1d27

// ─── Camera (Arc-Rotate — drag to orbit, scroll to zoom) ─────────────────────
const camera = new BABYLON.ArcRotateCamera(
  "cam",
  -Math.PI / 2,   // alpha: look from front
  Math.PI / 3,    // beta:  slightly above eye level
  3.5,            // radius
  new BABYLON.Vector3(0, 1.1, 0), // target: roughly the avatar's chest
  scene
);
camera.attachControl(canvas, true);
camera.lowerRadiusLimit  = 0.8;
camera.upperRadiusLimit  = 8;
camera.lowerBetaLimit    = 0.1;
camera.upperBetaLimit    = Math.PI / 1.8;
camera.wheelDeltaPercentage = 0.01;

// ─── Lighting ─────────────────────────────────────────────────────────────────
// Soft ambient dome
const hemi = new BABYLON.HemisphericLight(
  "hemi",
  new BABYLON.Vector3(0, 1, 0),
  scene
);
hemi.intensity      = 0.75;
hemi.groundColor    = new BABYLON.Color3(0.15, 0.15, 0.2);

// Key light (front-left, slightly high)
const keyLight = new BABYLON.DirectionalLight(
  "key",
  new BABYLON.Vector3(-0.5, -1, -0.8),
  scene
);
keyLight.intensity  = 1.2;
keyLight.position   = new BABYLON.Vector3(3, 6, 4);

// Rim / fill light (back-right)
const fillLight = new BABYLON.DirectionalLight(
  "fill",
  new BABYLON.Vector3(1, -0.5, 0.5),
  scene
);
fillLight.intensity = 0.4;
fillLight.diffuse   = new BABYLON.Color3(0.6, 0.7, 1.0); // cool blue fill

// ─── Ground grid (subtle reference plane) ─────────────────────────────────────
const ground = BABYLON.MeshBuilder.CreateGround(
  "ground",
  { width: 4, height: 4, subdivisions: 20 },
  scene
);
// Use GridMaterial if the materials library CDN script loaded it, else fallback.
let groundMat;
if (BABYLON.GridMaterial) {
  groundMat = new BABYLON.GridMaterial("gridMat", scene);
  groundMat.gridRatio           = 0.2;
  groundMat.majorUnitFrequency  = 5;
  groundMat.minorUnitVisibility = 0.2;
  groundMat.mainColor           = new BABYLON.Color3(0.18, 0.20, 0.28);
  groundMat.lineColor           = new BABYLON.Color3(0.25, 0.28, 0.38);
  groundMat.backFaceCulling     = false;
} else {
  groundMat = new BABYLON.StandardMaterial("groundMat", scene);
  groundMat.diffuseColor  = new BABYLON.Color3(0.15, 0.17, 0.22);
  groundMat.specularColor = BABYLON.Color3.Black();
}
ground.material = groundMat;

// ─── UI helpers (declared before first use) ───────────────────────────────────
const overlay = document.getElementById("status-overlay");

function setStatus(msg) {
  overlay.textContent   = msg;
  overlay.style.display = msg ? "block" : "none";
}

// ─── Avatar animator ──────────────────────────────────────────────────────────
const animator = new AvatarAnimator(scene, "assets/Untitled.glb");
setStatus("Loading avatar…");
animator.onReady = () => setStatus('Enter a sentence above and click "Sign it".');
animator.onError = (msg) => setStatus(`⚠ ${msg}`);

// ─── Render loop ──────────────────────────────────────────────────────────────
engine.runRenderLoop(() => {
  const delta = engine.getDeltaTime() / 1000; // ms → seconds
  animator.update(delta, currentSpeed());
  scene.render();
});

// ─── Resize ───────────────────────────────────────────────────────────────────
window.addEventListener("resize", () => engine.resize());

function currentSpeed() {
  return parseFloat(document.getElementById("speed-slider").value);
}

function showGloss(glossesUsed, glossesUnknown) {
  const container  = document.getElementById("gloss-display");
  const unknownSet = new Set(glossesUnknown);
  container.innerHTML = glossesUsed
    .map((g) => {
      const cls = unknownSet.has(g) ? "gloss-chip unknown" : "gloss-chip";
      return `<span class="${cls}">${g}</span>`;
    })
    .join("");
}

// ─── Sign button ──────────────────────────────────────────────────────────────
const signBtn   = document.getElementById("sign-btn");
const replayBtn = document.getElementById("replay-btn");
const textarea  = document.getElementById("input-text");

signBtn.addEventListener("click", async () => {
  const text = textarea.value.trim();
  if (!text) return;

  signBtn.disabled  = true;
  replayBtn.disabled = true;
  setStatus("⏳ Fetching animation…");
  document.getElementById("gloss-display").innerHTML = "";

  try {
    const animation = await fetchAnimation(text, { fps: 30 });
    animator.loadAnimation(animation, () => {
      setStatus("Done. Press Replay to watch again.");
      replayBtn.disabled = false;
    });
    showGloss(animation.glosses_used, animation.glosses_unknown);
    setStatus(""); // hide overlay while playing
  } catch (err) {
    setStatus(`❌ Error: ${err.message}`);
  } finally {
    signBtn.disabled = false;
  }
});

// Enter (without Shift) submits.
textarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    signBtn.click();
  }
});

// ─── Replay button ────────────────────────────────────────────────────────────
replayBtn.addEventListener("click", () => {
  animator.replay();
  setStatus("");
  replayBtn.disabled = true;
  const wait = setInterval(() => {
    if (!animator.isPlaying) {
      clearInterval(wait);
      replayBtn.disabled = false;
    }
  }, 200);
});

// ─── Speed slider ─────────────────────────────────────────────────────────────
const speedSlider = document.getElementById("speed-slider");
const speedLabel  = document.getElementById("speed-label");
speedSlider.addEventListener("input", () => {
  speedLabel.textContent = `${parseFloat(speedSlider.value)}×`;
});
