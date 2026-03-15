/**
 * avatar_animator.js
 *
 * Drives a 3-D stick-figure avatar from MediaPipe landmark data.
 *
 * The figure uses:
 *   • Cylinder meshes for limb segments (thick, easy to read)
 *   • Sphere meshes for joints
 *   • A sphere for the head
 *   • Color-coded sides: blue = left, orange = right, white = centre
 *   • Fingers drawn on each hand when landmarks are present
 *
 * If a GLB avatar URL is supplied the animator will also try to drive it
 * via FK bone rotations (optional upgrade path).
 */

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

// ─── MediaPipe pose landmark indices ────────────────────────────────────────
const MP = {
  NOSE: 0,
  LEFT_EYE: 2,         RIGHT_EYE: 5,
  LEFT_EAR: 7,         RIGHT_EAR: 8,
  LEFT_SHOULDER: 11,   RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,      RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,      RIGHT_WRIST: 16,
  LEFT_HIP: 23,        RIGHT_HIP: 24,
  LEFT_KNEE: 25,       RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,      RIGHT_ANKLE: 28,
};

// ─── Colour palette ──────────────────────────────────────────────────────────
const COLOR = {
  left:   0x4f8ef7,   // blue
  right:  0xf97316,   // orange
  centre: 0xe2e8f0,   // near-white
  head:   0xfbbf24,   // amber
  joint:  0xffffff,
  fingerLeft:  0x93c5fd,
  fingerRight: 0xfcd34d,
};

// ─── Segment definitions [fromIdx, toIdx, colorKey] ─────────────────────────
const POSE_SEGMENTS = [
  // Centre body
  [MP.LEFT_SHOULDER,  MP.RIGHT_SHOULDER, "centre"],
  [MP.LEFT_HIP,       MP.RIGHT_HIP,      "centre"],
  // Spine (we synthesise midpoints in code)
  // Left arm
  [MP.LEFT_SHOULDER,  MP.LEFT_ELBOW,     "left"],
  [MP.LEFT_ELBOW,     MP.LEFT_WRIST,     "left"],
  // Right arm
  [MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW,    "right"],
  [MP.RIGHT_ELBOW,    MP.RIGHT_WRIST,    "right"],
  // Torso sides
  [MP.LEFT_SHOULDER,  MP.LEFT_HIP,       "left"],
  [MP.RIGHT_SHOULDER, MP.RIGHT_HIP,      "right"],
  // Left leg
  [MP.LEFT_HIP,       MP.LEFT_KNEE,      "left"],
  [MP.LEFT_KNEE,      MP.LEFT_ANKLE,     "left"],
  // Right leg
  [MP.RIGHT_HIP,      MP.RIGHT_KNEE,     "right"],
  [MP.RIGHT_KNEE,     MP.RIGHT_ANKLE,    "right"],
];

// Hand finger chains (landmark index pairs)
const FINGER_CHAINS = [
  [0, 1, 2, 3, 4],       // thumb
  [0, 5, 6, 7, 8],       // index
  [0, 9, 10, 11, 12],    // middle
  [0, 13, 14, 15, 16],   // ring
  [0, 17, 18, 19, 20],   // pinky
];

// ─── Geometry helpers ────────────────────────────────────────────────────────

/** Create a cylinder mesh between two THREE.Vector3 points. */
function makeCylinder(color, radius = 0.04) {
  const mat = new THREE.MeshPhongMaterial({ color, shininess: 40 });
  const geo = new THREE.CylinderGeometry(radius, radius, 1, 8, 1);
  const mesh = new THREE.Mesh(geo, mat);
  return mesh;
}

/** Update an existing cylinder to stretch between points A and B. */
function updateCylinder(mesh, a, b) {
  const dir = new THREE.Vector3().subVectors(b, a);
  const len = dir.length();
  if (len < 1e-5) { mesh.visible = false; return; }
  mesh.visible = true;
  mesh.scale.y = len;
  mesh.position.copy(a).addScaledVector(dir.normalize(), len / 2);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
}

function makeSphere(color, radius = 0.055) {
  const mat = new THREE.MeshPhongMaterial({ color, shininess: 60 });
  const geo = new THREE.SphereGeometry(radius, 10, 10);
  return new THREE.Mesh(geo, mat);
}

// ─── Coordinate conversion ───────────────────────────────────────────────────
// MediaPipe: x right (0→1), y down (0→1), z depth relative to hip.
// Three.js: x right, y up, z toward viewer.
// We scale so the figure is roughly 2 units tall.
const SCALE = 3.5;

function mpToV3([x, y, z]) {
  return new THREE.Vector3(
    (x - 0.5) * SCALE,
    -(y - 0.5) * SCALE + 0.5,
    -z * SCALE * 0.3         // depth is compressed for readability
  );
}

// ─── Build the stick figure ──────────────────────────────────────────────────
function buildStickFigure(scene) {
  const group = new THREE.Group();
  scene.add(group);

  // Head sphere
  const head = makeSphere(COLOR.head, 0.14);
  group.add(head);

  // Spine cylinder (shoulder-midpoint → hip-midpoint)
  const spine = makeCylinder(COLOR.centre, 0.045);
  group.add(spine);

  // Pose limb cylinders
  const poseCylinders = POSE_SEGMENTS.map(([, , colorKey]) => {
    const cyl = makeCylinder(COLOR[colorKey]);
    group.add(cyl);
    return cyl;
  });

  // Joint spheres for key landmarks
  const KEY_JOINTS = [
    MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER,
    MP.LEFT_ELBOW,    MP.RIGHT_ELBOW,
    MP.LEFT_WRIST,    MP.RIGHT_WRIST,
    MP.LEFT_HIP,      MP.RIGHT_HIP,
    MP.LEFT_KNEE,     MP.RIGHT_KNEE,
    MP.LEFT_ANKLE,    MP.RIGHT_ANKLE,
  ];
  const jointSpheres = {};
  KEY_JOINTS.forEach((idx) => {
    const col = [MP.LEFT_SHOULDER,MP.LEFT_ELBOW,MP.LEFT_WRIST,MP.LEFT_HIP,MP.LEFT_KNEE,MP.LEFT_ANKLE].includes(idx)
      ? COLOR.left : COLOR.right;
    const s = makeSphere(col, 0.055);
    group.add(s);
    jointSpheres[idx] = s;
  });

  // Hand finger cylinders — 5 fingers × 4 segments × 2 hands = 40
  function buildHandCylinders(colorKey) {
    return FINGER_CHAINS.map((chain) =>
      chain.slice(0, -1).map(() => {
        const c = makeCylinder(COLOR[colorKey], 0.018);
        group.add(c);
        return c;
      })
    );
  }
  const leftFingers  = buildHandCylinders("fingerLeft");
  const rightFingers = buildHandCylinders("fingerRight");

  return { group, head, spine, poseCylinders, jointSpheres, leftFingers, rightFingers };
}

// ─── Main animator class ─────────────────────────────────────────────────────
export class AvatarAnimator {
  constructor(scene, avatarUrl = null) {
    this._scene    = scene;
    this._figure   = buildStickFigure(scene);
    this._frames   = [];
    this._currentFrame = 0;
    this._fps      = 30;
    this._playing  = false;
    this._elapsed  = 0;
    this._onComplete = null;

    // Optional GLB overlay (bones driven on top of stick figure)
    this._bones    = {};
    this._restQuats = {};
    if (avatarUrl) this._loadGLB(avatarUrl);
  }

  _loadGLB(url) {
    new GLTFLoader().load(url, (gltf) => {
      this._scene.add(gltf.scene);
      gltf.scene.traverse((obj) => {
        if (obj.isBone) {
          this._bones[obj.name] = obj;
          this._restQuats[obj.name] = obj.quaternion.clone();
        }
      });
    });
  }

  loadAnimation(animationData, onComplete = null) {
    this._frames   = animationData.frames ?? [];
    this._fps      = animationData.fps ?? 30;
    this._currentFrame = 0;
    this._elapsed  = 0;
    this._playing  = this._frames.length > 0;
    this._onComplete = onComplete;
  }

  stop()   { this._playing = false; }
  replay() { this._currentFrame = 0; this._elapsed = 0; this._playing = this._frames.length > 0; }

  get isPlaying() { return this._playing; }
  get progress()  { return this._frames.length ? this._currentFrame / this._frames.length : 0; }

  update(delta, speed = 1) {
    if (!this._playing || !this._frames.length) return;

    this._elapsed += delta * speed;
    this._currentFrame = Math.floor(this._elapsed * this._fps);

    if (this._currentFrame >= this._frames.length) {
      this._currentFrame = this._frames.length - 1;
      this._playing = false;
      if (this._onComplete) this._onComplete();
      return;
    }

    this._applyFrame(this._frames[this._currentFrame]);
  }

  _applyFrame(frame) {
    const pose = frame.pose;
    const lh   = frame.left_hand;
    const rh   = frame.right_hand;
    const { head, spine, poseCylinders, jointSpheres, leftFingers, rightFingers } = this._figure;

    if (!pose) return;

    // Convert all pose landmarks to Three.js vectors
    const pv = pose.map(mpToV3);

    // ── Head: position above nose using ear/shoulder scale ──────────────────
    if (pv[MP.NOSE]) {
      head.position.copy(pv[MP.NOSE]);
      // Estimate head radius from shoulder width
      const sw = pv[MP.LEFT_SHOULDER].distanceTo(pv[MP.RIGHT_SHOULDER]);
      const r = Math.max(0.10, Math.min(sw * 0.35, 0.20));
      head.scale.setScalar(r / 0.14);
    }

    // ── Spine ────────────────────────────────────────────────────────────────
    const shoulderMid = new THREE.Vector3()
      .addVectors(pv[MP.LEFT_SHOULDER], pv[MP.RIGHT_SHOULDER]).multiplyScalar(0.5);
    const hipMid = new THREE.Vector3()
      .addVectors(pv[MP.LEFT_HIP], pv[MP.RIGHT_HIP]).multiplyScalar(0.5);
    updateCylinder(spine, hipMid, shoulderMid);

    // ── Pose limb segments ───────────────────────────────────────────────────
    POSE_SEGMENTS.forEach(([fromIdx, toIdx], i) => {
      updateCylinder(poseCylinders[i], pv[fromIdx], pv[toIdx]);
    });

    // ── Joint spheres ────────────────────────────────────────────────────────
    Object.entries(jointSpheres).forEach(([idx, mesh]) => {
      if (pv[idx]) mesh.position.copy(pv[idx]);
    });

    // ── Hands ────────────────────────────────────────────────────────────────
    function applyHand(landmarks, fingerCyls) {
      if (!landmarks) {
        fingerCyls.forEach((chain) => chain.forEach((c) => (c.visible = false)));
        return;
      }
      const hv = landmarks.map(mpToV3);
      FINGER_CHAINS.forEach((chain, fi) => {
        for (let si = 0; si < chain.length - 1; si++) {
          updateCylinder(fingerCyls[fi][si], hv[chain[si]], hv[chain[si + 1]]);
        }
      });
    }

    applyHand(lh, leftFingers);
    applyHand(rh, rightFingers);
  }
}
