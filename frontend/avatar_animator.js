/**
 * avatar_animator.js — Babylon.js + Ready Player Me
 *
 * Drives a Ready Player Me (Mixamo-rigged) avatar from per-frame MediaPipe
 * skeleton data delivered by the FastAPI backend.
 *
 * Technique breakdown
 * ───────────────────
 * Arms      — BoneIKController: wrist target + elbow pole target mesh are
 *             repositioned each frame, then ikController.update() solves the
 *             upper-arm → forearm chain automatically.
 *
 * Wrist     — After IK positions the forearm, we set the wrist bone rotation
 *             to match the hand plane (palm normal from MediaPipe hand landmarks).
 *
 * Fingers   — FK: compute the curl angle at each knuckle from MediaPipe hand
 *             landmark triplets and apply it as a local Z-axis rotation on
 *             each finger bone.
 *
 * Spine     — FK: derive a small lean/tilt from the hip-midpoint →
 *             shoulder-midpoint direction and apply to the Spine2 bone.
 *
 * Head      — FK: tilt toward the nose landmark.
 *
 * Fallback  — If avatar.glb is not found, a simple Babylon.js stick figure
 *             (LinesMesh) is displayed so the app still demonstrates motion.
 *
 * Ready Player Me export settings required
 * ─────────────────────────────────────────
 * Pose      : T-Pose
 * Meshes    : All (or Body + Outfit)
 * Hand pose : MUST include finger bones
 * Format    : GLB
 *
 * BABYLON is a global loaded by the CDN <script> in index.html.
 */

// ─── MediaPipe pose landmark indices ─────────────────────────────────────────
const MP = {
  NOSE:            0,
  LEFT_EYE:        2,  RIGHT_EYE:       5,
  LEFT_SHOULDER:  11,  RIGHT_SHOULDER: 12,
  LEFT_ELBOW:     13,  RIGHT_ELBOW:    14,
  LEFT_WRIST:     15,  RIGHT_WRIST:    16,
  LEFT_HIP:       23,  RIGHT_HIP:      24,
  LEFT_KNEE:      25,  RIGHT_KNEE:     26,
  LEFT_ANKLE:     27,  RIGHT_ANKLE:    28,
};

// ─── RPM / Mixamo bone names ─────────────────────────────────────────────────
// Ready Player Me uses standard Mixamo naming (no "mixamorig:" prefix).
const BONES = {
  hips:         "Hips",
  spine:        "Spine",
  spine1:       "Spine1",
  spine2:       "Spine2",
  neck:         "Neck",
  head:         "Head",
  leftShoulder: "LeftShoulder",  rightShoulder: "RightShoulder",
  leftArm:      "LeftArm",       rightArm:      "RightArm",       // upper arm
  leftForeArm:  "LeftForeArm",   rightForeArm:  "RightForeArm",
  leftHand:     "LeftHand",      rightHand:     "RightHand",
  leftUpLeg:    "LeftUpLeg",     rightUpLeg:    "RightUpLeg",
  leftLeg:      "LeftLeg",       rightLeg:      "RightLeg",
};

// Finger bone chains per side.
// Each entry: [bone1, bone2, bone3] = proximal → middle → distal
const FINGER_BONES = {
  left: {
    thumb:  ["LeftHandThumb1",  "LeftHandThumb2",  "LeftHandThumb3"],
    index:  ["LeftHandIndex1",  "LeftHandIndex2",  "LeftHandIndex3"],
    middle: ["LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3"],
    ring:   ["LeftHandRing1",   "LeftHandRing2",   "LeftHandRing3"],
    pinky:  ["LeftHandPinky1",  "LeftHandPinky2",  "LeftHandPinky3"],
  },
  right: {
    thumb:  ["RightHandThumb1",  "RightHandThumb2",  "RightHandThumb3"],
    index:  ["RightHandIndex1",  "RightHandIndex2",  "RightHandIndex3"],
    middle: ["RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3"],
    ring:   ["RightHandRing1",   "RightHandRing2",   "RightHandRing3"],
    pinky:  ["RightHandPinky1",  "RightHandPinky2",  "RightHandPinky3"],
  },
};

// MediaPipe hand landmark indices per finger (proximal→tip).
const FINGER_LM = {
  thumb:  [1, 2, 3, 4],
  index:  [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring:   [13, 14, 15, 16],
  pinky:  [17, 18, 19, 20],
};

// ─── Coordinate conversion ────────────────────────────────────────────────────
// MediaPipe: x ∈ [0,1] (left→right), y ∈ [0,1] (top→bottom), z depth.
// Babylon.js: x right, y up, z forward (left-handed).
//
// We mirror X so the avatar faces the viewer (signer's right = viewer's right).
const SCALE  = 2.0;   // world units for a ~2m tall avatar
const HIP_Y  = 1.0;   // approximate world-y of the Hips bone in T-pose

function poseToV3([x, y, z]) {
  return new BABYLON.Vector3(
    -(x - 0.5) * SCALE,            // mirror X
    -(y - 0.5) * SCALE + HIP_Y,    // flip Y, shift up
    -z * SCALE * 0.25,             // depth compressed for readability
  );
}

// ─── Quaternion helper: rotate vector A to align with vector B ───────────────
function quatFromVecToVec(from, to) {
  from = from.clone().normalize();
  to   = to.clone().normalize();
  const dot  = BABYLON.Vector3.Dot(from, to);
  const axis = BABYLON.Vector3.Cross(from, to);
  if (axis.lengthSquared() < 1e-10) {
    // Vectors are parallel — identity or 180° flip
    return dot > 0
      ? BABYLON.Quaternion.Identity()
      : BABYLON.Quaternion.RotationAxis(
          BABYLON.Vector3.Right(), Math.PI
        );
  }
  const angle = Math.acos(Math.max(-1, Math.min(1, dot)));
  return BABYLON.Quaternion.RotationAxis(axis.normalize(), angle);
}

// ─── Finger curl angle between three landmarks ────────────────────────────────
function curlAngle(a, b, c) {
  const v1 = b.subtract(a).normalize();
  const v2 = c.subtract(b).normalize();
  return Math.acos(Math.max(-1, Math.min(1, BABYLON.Vector3.Dot(v1, v2))));
}

// ─── Build the fallback stick figure (shown when avatar.glb is absent) ───────
function buildStickFigure(scene) {
  const color = new BABYLON.Color3(0.31, 0.56, 0.97); // accent blue

  function makeCyl(name) {
    const m = BABYLON.MeshBuilder.CreateCylinder(
      name, { diameter: 0.06, height: 1, tessellation: 8 }, scene
    );
    const mat = new BABYLON.StandardMaterial(name + "_mat", scene);
    mat.diffuseColor = color;
    m.material = mat;
    return m;
  }
  function makeSphere(name, d = 0.1) {
    const m = BABYLON.MeshBuilder.CreateSphere(name, { diameter: d }, scene);
    const mat = new BABYLON.StandardMaterial(name + "_mat", scene);
    mat.diffuseColor = new BABYLON.Color3(1.0, 0.75, 0.15);
    m.material = mat;
    return m;
  }

  const head   = makeSphere("sf_head", 0.25);
  const spine  = makeCyl("sf_spine");
  const lUpper = makeCyl("sf_lu");
  const lLower = makeCyl("sf_ll");
  const rUpper = makeCyl("sf_ru");
  const rLower = makeCyl("sf_rl");

  function updateCyl(mesh, a, b) {
    const dir = b.subtract(a);
    const len = dir.length();
    if (len < 1e-5) { mesh.isVisible = false; return; }
    mesh.isVisible = true;
    mesh.scaling.y = len;
    mesh.position  = BABYLON.Vector3.Lerp(a, b, 0.5);
    const up = new BABYLON.Vector3(0, 1, 0);
    const q  = quatFromVecToVec(up, dir.normalize());
    mesh.rotationQuaternion = q;
  }

  function applyStick(frame) {
    if (!frame?.pose) return;
    const p = frame.pose.map(poseToV3);
    head.position = p[MP.NOSE];
    const shoulderMid = BABYLON.Vector3.Lerp(p[MP.LEFT_SHOULDER],  p[MP.RIGHT_SHOULDER], 0.5);
    const hipMid      = BABYLON.Vector3.Lerp(p[MP.LEFT_HIP],       p[MP.RIGHT_HIP],      0.5);
    updateCyl(spine,  hipMid,               shoulderMid);
    updateCyl(lUpper, p[MP.LEFT_SHOULDER],  p[MP.LEFT_ELBOW]);
    updateCyl(lLower, p[MP.LEFT_ELBOW],     p[MP.LEFT_WRIST]);
    updateCyl(rUpper, p[MP.RIGHT_SHOULDER], p[MP.RIGHT_ELBOW]);
    updateCyl(rLower, p[MP.RIGHT_ELBOW],    p[MP.RIGHT_WRIST]);
  }

  function dispose() {
    [head, spine, lUpper, lLower, rUpper, rLower].forEach(m => m.dispose());
  }

  return { applyStick, dispose };
}

// ─── Main class ───────────────────────────────────────────────────────────────
export class AvatarAnimator {
  constructor(scene, avatarUrl) {
    this._scene      = scene;
    this._frames     = [];
    this._fps        = 30;
    this._elapsed    = 0;
    this._playing    = false;
    this._onComplete = null;

    // Babylon.js skeleton + mesh (set after GLB loads)
    this._skeleton   = null;
    this._mesh       = null;     // the skinned mesh
    this._bones      = {};       // name → Bone
    this._bindDirs   = {};       // name → world-space "Y-up" direction at bind time

    // IK target meshes (invisible, repositioned each frame)
    this._ikTargets  = {};       // populated in _setupIK()

    // Fallback stick figure (shown while / if avatar fails to load)
    this._stick      = null;

    // Callbacks wired by main.js
    this.onReady     = null;
    this.onError     = null;

    this._loadAvatar(avatarUrl);
  }

  // ── Load GLB ───────────────────────────────────────────────────────────────
  async _loadAvatar(url) {
    try {
      const dir  = url.substring(0, url.lastIndexOf("/") + 1) || "./";
      const file = url.substring(url.lastIndexOf("/") + 1);
      const result = await BABYLON.SceneLoader.ImportMeshAsync(
        "", dir, file, this._scene
      );

      // Find the skinned mesh (the one that has a skeleton)
      const skinnedMesh = result.meshes.find(m => m.skeleton) ?? result.meshes[1];
      if (!skinnedMesh?.skeleton) throw new Error("No skeleton found in GLB.");

      this._mesh     = skinnedMesh;
      this._skeleton = skinnedMesh.skeleton;

      // Index bones by name
      this._skeleton.bones.forEach(b => {
        this._bones[b.name] = b;
      });

      // Capture bind-pose world directions BEFORE any animation is applied.
      // For Mixamo bones, the "bone direction" is the world-space Y axis of the
      // bone's absolute matrix — it points from the joint toward its child.
      this._skeleton.computeAbsoluteTransforms();
      this._skeleton.bones.forEach(b => {
        const m   = b.getAbsoluteInverseBindMatrix
          ? b.getAbsoluteInverseBindMatrix()  // preferred
          : b.getWorldMatrix();
        // Column 1 (rows 4-6) = world-space Y axis
        this._bindDirs[b.name] = new BABYLON.Vector3(m.m[4], m.m[5], m.m[6]).normalize();
      });

      this._setupIK();

      // Remove the stick figure once the real avatar is ready
      if (this._stick) {
        this._stick.dispose();
        this._stick = null;
      }

      if (this.onReady) this.onReady();
    } catch (err) {
      console.warn("[AvatarAnimator] GLB load failed:", err.message);
      // Show stick figure as fallback
      this._stick = buildStickFigure(this._scene);
      if (this.onError) {
        this.onError(
          "avatar.glb not found. Using stick-figure fallback.\n" +
          "Convert your Mixamo FBX to GLB (use Blender: File→Export→glTF 2.0) and save to  frontend/assets/avatar.glb"
        );
      }
    }
  }

  // ── Set up BoneIKController for both arms ──────────────────────────────────
  _setupIK() {
    const scene = this._scene;

    // Helper: invisible sphere used as an IK target or pole target
    const makeTarget = (name) => {
      const m = BABYLON.MeshBuilder.CreateSphere(name, { diameter: 0.04 }, scene);
      m.isVisible = false;
      return m;
    };

    const sides = [
      { side: "left",  foreArmBone: BONES.leftForeArm,  wristIdx: MP.LEFT_WRIST,  elbowIdx: MP.LEFT_ELBOW  },
      { side: "right", foreArmBone: BONES.rightForeArm, wristIdx: MP.RIGHT_WRIST, elbowIdx: MP.RIGHT_ELBOW },
    ];

    for (const { side, foreArmBone, wristIdx, elbowIdx } of sides) {
      const bone = this._bones[foreArmBone];
      if (!bone) continue;

      const wristTarget = makeTarget(`ik_wrist_${side}`);
      const elbowPole   = makeTarget(`ik_elbow_${side}`);

      const ik = new BABYLON.BoneIKController(this._mesh, bone, {
        targetMesh:     wristTarget,
        poleTargetMesh: elbowPole,
        // poleAngle flips which side the elbow bends toward.
        // Left arm bends outward (Math.PI), right arm inward (0) in Mixamo.
        poleAngle:  side === "left" ? Math.PI : 0,
        maxAngle:   Math.PI * 0.92,
        slerpAmount: 1,            // no lag — we handle smoothing at the frame level
      });

      this._ikTargets[side] = { wristTarget, elbowPole, ik, wristIdx, elbowIdx };
    }
  }

  // ── Public API ─────────────────────────────────────────────────────────────
  loadAnimation(animationData, onComplete = null) {
    this._frames     = animationData.frames ?? [];
    this._fps        = animationData.fps    ?? 30;
    this._elapsed    = 0;
    this._playing    = this._frames.length > 0;
    this._onComplete = onComplete;
  }

  stop()   { this._playing = false; }
  replay() { this._elapsed = 0; this._playing = this._frames.length > 0; }

  get isPlaying() { return this._playing; }
  get progress()  { return this._frames.length ? Math.min(this._elapsed * this._fps / this._frames.length, 1) : 0; }

  update(delta, speed = 1) {
    if (!this._playing || !this._frames.length) return;

    this._elapsed += delta * speed;
    const idx      = Math.floor(this._elapsed * this._fps);

    if (idx >= this._frames.length) {
      this._applyFrame(this._frames[this._frames.length - 1]);
      this._playing = false;
      if (this._onComplete) this._onComplete();
      return;
    }

    this._applyFrame(this._frames[idx]);
  }

  // ── Apply a single frame ───────────────────────────────────────────────────
  _applyFrame(frame) {
    if (!frame?.pose) return;

    // Stick-figure fallback path
    if (this._stick) {
      this._stick.applyStick(frame);
      return;
    }

    if (!this._skeleton) return;

    const pv = frame.pose.map(poseToV3);

    this._driveArms(pv);
    this._driveSpineHead(pv);

    if (frame.left_hand)  this._driveFingers(frame.left_hand,  "left",  pv[MP.LEFT_WRIST]);
    if (frame.right_hand) this._driveFingers(frame.right_hand, "right", pv[MP.RIGHT_WRIST]);
  }

  // ── Arms: BoneIKController ─────────────────────────────────────────────────
  _driveArms(pv) {
    for (const { wristTarget, elbowPole, ik, wristIdx, elbowIdx } of Object.values(this._ikTargets)) {
      wristTarget.position = pv[wristIdx];
      elbowPole.position   = pv[elbowIdx];
      ik.update();
    }
  }

  // ── Spine + Head: FK from pose landmarks ──────────────────────────────────
  _driveSpineHead(pv) {
    // Derive lean/tilt from hip-midpoint → shoulder-midpoint vector
    const hipMid      = BABYLON.Vector3.Lerp(pv[MP.LEFT_HIP],      pv[MP.RIGHT_HIP],      0.5);
    const shoulderMid = BABYLON.Vector3.Lerp(pv[MP.LEFT_SHOULDER],  pv[MP.RIGHT_SHOULDER], 0.5);

    const spineDir  = shoulderMid.subtract(hipMid).normalize();
    const restSpine = new BABYLON.Vector3(0, 1, 0); // T-pose is straight up

    const spine2Bone = this._bones[BONES.spine2];
    if (spine2Bone) {
      const q = quatFromVecToVec(restSpine, spineDir);
      spine2Bone.setRotationQuaternion(q, BABYLON.Space.WORLD, this._mesh);
    }

    // Head: tilt slightly toward nose landmark
    const headBone = this._bones[BONES.head];
    if (headBone && pv[MP.NOSE]) {
      const headDir = pv[MP.NOSE].subtract(shoulderMid).normalize();
      const q = quatFromVecToVec(restSpine, headDir);
      headBone.setRotationQuaternion(q, BABYLON.Space.WORLD, this._mesh);
    }
  }

  // ── Fingers: FK from hand landmarks ───────────────────────────────────────
  // Converts MediaPipe hand landmarks to world-space positions relative to the
  // wrist, then computes joint curl angles and applies them as local Z-axis
  // rotations on each finger bone.
  //
  // Curl axis: For Mixamo/RPM rigs, fingers flex around the bone's local Z-axis.
  // Left hand  → negative Z = curl toward palm.
  // Right hand → positive Z = curl toward palm.
  // Adjust the curlSign constant if the fingers curl the wrong way.
  _driveFingers(handLandmarks, side, wristWorldPos) {
    const curlSign = side === "left" ? -1 : 1;
    const boneSide = FINGER_BONES[side];

    // Transform hand landmarks to world space.
    // hand[0] is the wrist, already positioned by IK.
    // The other landmarks are in the same normalized image space as pose landmarks,
    // so the same poseToV3 transform gives their world-space positions.
    const hv = handLandmarks.map(poseToV3);

    // Anchor: correct the wrist position to match the IK-solved wrist.
    // (MediaPipe wrist may differ slightly from the IK result.)
    const wristOffset = wristWorldPos.subtract(hv[0]);
    const hwv = hv.map(v => v.add(wristOffset));

    for (const [fingerName, lmIndices] of Object.entries(FINGER_LM)) {
      const boneNames = boneSide[fingerName];
      if (!boneNames) continue;

      // lmIndices = [mcp, pip, dip, tip]  (4 landmarks, 3 bones)
      for (let i = 0; i < boneNames.length; i++) {
        const bone = this._bones[boneNames[i]];
        if (!bone) continue;

        const a = hwv[lmIndices[i]];
        const b = hwv[lmIndices[i + 1]];
        const c = hwv[lmIndices[i + 2]];

        const curl = curlAngle(a, b, c);  // 0 = straight, π = fully curled

        // Apply curl as local Z rotation.
        // We preserve any existing X/Y rotation (from IK or prior frames).
        const current = bone.rotationQuaternion
          ?? BABYLON.Quaternion.FromEulerAngles(
               bone.rotation.x, bone.rotation.y, bone.rotation.z
             );
        const euler    = current.toEulerAngles();
        const newQ     = BABYLON.Quaternion.FromEulerAngles(
          euler.x,
          euler.y,
          curlSign * curl
        );
        bone.setRotationQuaternion(newQ, BABYLON.Space.LOCAL);
      }
    }
  }
}
