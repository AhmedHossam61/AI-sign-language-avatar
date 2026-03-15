/**
 * main.js — Three.js scene setup + UI wiring.
 *
 * Ties together:
 *   - Three.js renderer / camera / lighting
 *   - AvatarAnimator (avatar_animator.js)
 *   - fetchAnimation (api_client.js)
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { AvatarAnimator } from "./avatar_animator.js";
import { fetchAnimation } from "./api_client.js";

// ─── Scene setup ─────────────────────────────────────────────────────────────
const canvas    = document.getElementById("three-canvas");
const container = document.getElementById("canvas-container");
const overlay   = document.getElementById("status-overlay");

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x1a1d27);

const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
camera.position.set(0, 1.2, 3.5);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 1.0, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(2, 4, 3);
scene.add(dirLight);

// Ground grid (subtle reference plane)
const grid = new THREE.GridHelper(4, 20, 0x2e3347, 0x2e3347);
grid.position.y = -0.01;
scene.add(grid);

// ─── Avatar animator ─────────────────────────────────────────────────────────
// To use a GLB avatar: pass the URL as the second argument, e.g.
//   const animator = new AvatarAnimator(scene, "assets/avatar.glb");
// For now we use the wire-frame fallback (null → no GLB needed).
const animator = new AvatarAnimator(scene, null);

setStatus("Enter a sentence above and click \"Sign it\".");

// ─── Resize handler ───────────────────────────────────────────────────────────
function resize() {
  const w = container.clientWidth;
  const h = container.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
resize();
new ResizeObserver(resize).observe(container);

// ─── Render loop ──────────────────────────────────────────────────────────────
const clock = new THREE.Clock();

(function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();
  controls.update();
  animator.update(delta, currentSpeed());
  renderer.render(scene, camera);
})();

// ─── UI helpers ───────────────────────────────────────────────────────────────
function setStatus(msg) {
  overlay.textContent = msg;
  overlay.style.display = msg ? "block" : "none";
}

function currentSpeed() {
  return parseFloat(document.getElementById("speed-slider").value);
}

function showGloss(glossesUsed, glossesUnknown) {
  const container = document.getElementById("gloss-display");
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

  signBtn.disabled = true;
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

// Allow pressing Enter (without Shift) to submit.
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
  // Re-enable replay when done.
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
