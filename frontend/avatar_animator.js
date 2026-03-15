/**
 * avatar_animator.js
 *
 * Drives a Three.js skeleton from MediaPipe landmark data.
 *
 * MediaPipe gives us 3-D POSITIONS for each joint.  To drive a rigged avatar
 * we need ROTATIONS.  For MVP we use a simple two-bone FK approach:
 *   • For each limb segment (upper-arm, lower-arm, etc.) we compute the
 *     direction vector between the two endpoint landmarks and rotate the bone
 *     so its local +Y axis aligns with that direction.
 *   • Hand finger joints are applied similarly.
 *
 * The avatar is expected to be a Ready Player Me (or Mixamo) GLB with standard
 * bone names (Hips, Spine, Spine1, Spine2, Neck, Head,
 * LeftUpperArm, LeftLowerArm, LeftHand, RightUpperArm, …).
 *
 * If no GLB is loaded the animator falls back to drawing the skeleton as a
 * THREE.SkeletonHelper wire-frame so you can verify data without an avatar.
 */

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

// ─── MediaPipe pose landmark indices ────────────────────────────────────────
const MP = {
  NOSE: 0,
  LEFT_SHOULDER: 11,  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,     RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,     RIGHT_WRIST: 16,
  LEFT_HIP: 23,       RIGHT_HIP: 24,
  LEFT_KNEE: 25,      RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,     RIGHT_ANKLE: 28,
};

// ─── MediaPipe hand landmark indices ────────────────────────────────────────
const MH = {
  WRIST: 0,
  INDEX_MCP: 5, INDEX_PIP: 6,  INDEX_DIP: 7,  INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13,  RING_PIP: 14,  RING_DIP: 15,  RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
  THUMB_CMC: 1,  THUMB_MCP: 2,  THUMB_IP: 3,   THUMB_TIP: 4,
};

// Bone name → [parent landmark index, child landmark index, hand side]
// "side" is null for pose bones, "left"/"right" for hand bones.
const BONE_SEGMENTS = [
  // Torso
  { bone: "Spine",        from: [MP.LEFT_HIP, MP.RIGHT_HIP], to: [MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER], midpoint: true },
  // Arms
  { bone: "LeftUpperArm",  from: MP.LEFT_SHOULDER,  to: MP.LEFT_ELBOW },
  { bone: "LeftLowerArm",  from: MP.LEFT_ELBOW,     to: MP.LEFT_WRIST },
  { bone: "RightUpperArm", from: MP.RIGHT_SHOULDER, to: MP.RIGHT_ELBOW },
  { bone: "RightLowerArm", from: MP.RIGHT_ELBOW,    to: MP.RIGHT_WRIST },
  // Legs
  { bone: "LeftUpperLeg",  from: MP.LEFT_HIP,       to: MP.LEFT_KNEE },
  { bone: "LeftLowerLeg",  from: MP.LEFT_KNEE,       to: MP.LEFT_ANKLE },
  { bone: "RightUpperLeg", from: MP.RIGHT_HIP,      to: MP.RIGHT_KNEE },
  { bone: "RightLowerLeg", from: MP.RIGHT_KNEE,      to: MP.RIGHT_ANKLE },
];

// Helper: average of multiple landmark positions.
function midpoint(positions) {
  const sum = positions.reduce(
    (acc, p) => [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]],
    [0, 0, 0]
  );
  return sum.map((v) => v / positions.length);
}

// Rotate bone so its +Y local axis points along `direction`.
function alignBoneToDirection(bone, direction, restQuaternion) {
  const dir = new THREE.Vector3(...direction).normalize();
  if (dir.lengthSq() < 1e-6) return;

  // Bone rest orientation: +Y is "along the limb".
  const up = new THREE.Vector3(0, 1, 0);
  const q = new THREE.Quaternion().setFromUnitVectors(up, dir);

  // Compose with rest quaternion so we respect the rig's bind pose.
  bone.quaternion.copy(restQuaternion).premultiply(q);
}

// ─── Fallback dot/line skeleton renderer (no GLB needed) ────────────────────
function buildFallbackSkeleton(scene) {
  const joints = {};
  const geo = new THREE.SphereGeometry(0.02, 6, 6);
  const mat = new THREE.MeshBasicMaterial({ color: 0x4f8ef7 });

  // 33 pose joints + 21 left + 21 right
  const totalJoints = 33 + 21 + 21;
  for (let i = 0; i < totalJoints; i++) {
    const mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh);
    joints[i] = mesh;
  }

  const lineMat = new THREE.LineBasicMaterial({ color: 0x8892a4 });
  const connections = [
    [MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER],
    [MP.LEFT_SHOULDER, MP.LEFT_ELBOW],
    [MP.LEFT_ELBOW, MP.LEFT_WRIST],
    [MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW],
    [MP.RIGHT_ELBOW, MP.RIGHT_WRIST],
    [MP.LEFT_HIP, MP.RIGHT_HIP],
    [MP.LEFT_HIP, MP.LEFT_KNEE],
    [MP.LEFT_KNEE, MP.LEFT_ANKLE],
    [MP.RIGHT_HIP, MP.RIGHT_KNEE],
    [MP.RIGHT_KNEE, MP.RIGHT_ANKLE],
    [MP.LEFT_SHOULDER, MP.LEFT_HIP],
    [MP.RIGHT_SHOULDER, MP.RIGHT_HIP],
  ];

  const lines = connections.map(([a, b]) => {
    const points = [new THREE.Vector3(), new THREE.Vector3()];
    const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
    const line = new THREE.Line(lineGeo, lineMat);
    scene.add(line);
    return { line, a, b };
  });

  return { joints, lines };
}

// ─── Main animator class ─────────────────────────────────────────────────────
export class AvatarAnimator {
  /**
   * @param {THREE.Scene} scene
   * @param {string|null} avatarUrl  URL to a GLB avatar, or null for wire-frame.
   */
  constructor(scene, avatarUrl = null) {
    this._scene = scene;
    this._bones = {};           // boneName → THREE.Bone
    this._restQuats = {};       // boneName → original quaternion clone
    this._fallback = null;      // fallback skeleton meshes
    this._frames = [];
    this._currentFrame = 0;
    this._fps = 30;
    this._playing = false;
    this._elapsed = 0;
    this._onComplete = null;

    if (avatarUrl) {
      this._loadGLB(avatarUrl);
    } else {
      this._fallback = buildFallbackSkeleton(scene);
    }
  }

  _loadGLB(url) {
    const loader = new GLTFLoader();
    loader.load(
      url,
      (gltf) => {
        this._scene.add(gltf.scene);
        gltf.scene.traverse((obj) => {
          if (obj.isBone || (obj.isObject3D && obj.name)) {
            this._bones[obj.name] = obj;
            this._restQuats[obj.name] = obj.quaternion.clone();
          }
        });
      },
      undefined,
      (err) => console.warn("GLB load error:", err)
    );
  }

  /** Load a new animation and start playing. */
  loadAnimation(animationData, onComplete = null) {
    this._frames = animationData.frames ?? [];
    this._fps = animationData.fps ?? 30;
    this._currentFrame = 0;
    this._elapsed = 0;
    this._playing = this._frames.length > 0;
    this._onComplete = onComplete;
  }

  stop() {
    this._playing = false;
  }

  replay() {
    this._currentFrame = 0;
    this._elapsed = 0;
    this._playing = this._frames.length > 0;
  }

  get isPlaying() { return this._playing; }
  get progress() {
    if (!this._frames.length) return 0;
    return this._currentFrame / this._frames.length;
  }

  /**
   * Call every animation frame from the render loop.
   * @param {number} delta  seconds since last frame (from THREE.Clock)
   * @param {number} speed  playback multiplier (1 = normal)
   */
  update(delta, speed = 1) {
    if (!this._playing || !this._frames.length) return;

    this._elapsed += delta * speed;
    const frameFloat = this._elapsed * this._fps;
    this._currentFrame = Math.floor(frameFloat);

    if (this._currentFrame >= this._frames.length) {
      this._currentFrame = this._frames.length - 1;
      this._playing = false;
      if (this._onComplete) this._onComplete();
      return;
    }

    this._applyFrame(this._frames[this._currentFrame]);
  }

  _applyFrame(frame) {
    if (this._fallback) {
      this._applyFallback(frame);
    } else {
      this._applyBones(frame);
    }
  }

  // ── Fallback wire-frame renderer ─────────────────────────────────────────
  _applyFallback(frame) {
    const pose = frame.pose;
    const lh   = frame.left_hand;
    const rh   = frame.right_hand;
    const { joints, lines } = this._fallback;

    // MediaPipe coords: x right, y down, z toward camera (NDC-ish, ~0-1).
    // Map to Three.js: x right, y up, z toward viewer.
    const toV3 = ([x, y, z]) => new THREE.Vector3(
      (x - 0.5) * 2,
      -(y - 0.5) * 2,
      -z
    );

    if (pose) {
      pose.forEach((lm, i) => {
        if (joints[i]) joints[i].position.copy(toV3(lm));
      });
    }

    // Hand joints: offset 33 (left) and 54 (right).
    if (lh) lh.forEach((lm, i) => {
      const j = joints[33 + i];
      if (j) j.position.copy(toV3(lm));
    });
    if (rh) rh.forEach((lm, i) => {
      const j = joints[54 + i];
      if (j) j.position.copy(toV3(lm));
    });

    // Update line geometries.
    lines.forEach(({ line, a, b }) => {
      if (!pose || !pose[a] || !pose[b]) return;
      const positions = line.geometry.attributes.position;
      const va = toV3(pose[a]);
      const vb = toV3(pose[b]);
      positions.setXYZ(0, va.x, va.y, va.z);
      positions.setXYZ(1, vb.x, vb.y, vb.z);
      positions.needsUpdate = true;
    });
  }

  // ── GLB bone driver ──────────────────────────────────────────────────────
  _applyBones(frame) {
    const pose = frame.pose;
    if (!pose) return;

    for (const seg of BONE_SEGMENTS) {
      const bone = this._bones[seg.bone];
      if (!bone) continue;

      let fromPos, toPos;
      if (seg.midpoint) {
        fromPos = midpoint(
          (Array.isArray(seg.from[0]) ? seg.from : [seg.from]).map((i) => pose[i])
        );
        toPos = midpoint(
          (Array.isArray(seg.to[0]) ? seg.to : [seg.to]).map((i) => pose[i])
        );
      } else {
        fromPos = pose[seg.from];
        toPos   = pose[seg.to];
      }

      if (!fromPos || !toPos) continue;
      const dir = [toPos[0] - fromPos[0], -(toPos[1] - fromPos[1]), toPos[2] - fromPos[2]];
      alignBoneToDirection(bone, dir, this._restQuats[seg.bone] ?? new THREE.Quaternion());
    }

    // Hips position (root translation).
    const hipsBone = this._bones["Hips"] ?? this._bones["mixamorigHips"];
    if (hipsBone && pose[MP.LEFT_HIP] && pose[MP.RIGHT_HIP]) {
      const hp = midpoint([pose[MP.LEFT_HIP], pose[MP.RIGHT_HIP]]);
      hipsBone.position.set((hp[0] - 0.5) * 2, -(hp[1] - 0.5) * 2 + 1, -hp[2]);
    }
  }
}
