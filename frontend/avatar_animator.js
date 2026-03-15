/**
 * avatar_animator.js — Babylon.js FK bone driver
 *
 * Drives a Mixamo-rigged avatar from per-frame MediaPipe pose/hand data
 * using pure Forward Kinematics (FK).
 *
 * For each limb bone we:
 *   1. Compute the world-space direction from parent joint → child joint
 *      using the MediaPipe landmark positions.
 *   2. Find the rotation that turns the bone's rest direction into that
 *      target direction.
 *   3. Apply it as a LOCAL rotation on the bone.
 *
 * BABYLON is a global loaded by the CDN <script> in index.html.
 */

// ─── MediaPipe pose landmark indices ─────────────────────────────────────────
const MP = {
  NOSE:            0,
  LEFT_SHOULDER:  11,  RIGHT_SHOULDER: 12,
  LEFT_ELBOW:     13,  RIGHT_ELBOW:    14,
  LEFT_WRIST:     15,  RIGHT_WRIST:    16,
  LEFT_HIP:       23,  RIGHT_HIP:      24,
  LEFT_KNEE:      25,  RIGHT_KNEE:     26,
  LEFT_ANKLE:     27,  RIGHT_ANKLE:    28,
};

// ─── Bone names (may get a prefix prepended at runtime) ──────────────────────
const BONES = {
  hips:           "Hips",
  spine:          "Spine",
  spine1:         "Spine1",
  spine2:         "Spine2",
  neck:           "Neck",
  head:           "Head",
  leftArm:        "LeftArm",
  leftForeArm:    "LeftForeArm",
  leftHand:       "LeftHand",
  rightArm:       "RightArm",
  rightForeArm:   "RightForeArm",
  rightHand:      "RightHand",
  leftUpLeg:      "LeftUpLeg",
  leftLeg:        "LeftLeg",
  rightUpLeg:     "RightUpLeg",
  rightLeg:       "RightLeg",
};

// FK chain: [boneName key, fromLandmark, toLandmark]
const FK_CHAINS = [
  ["leftArm",      MP.LEFT_SHOULDER,  MP.LEFT_ELBOW],
  ["leftForeArm",  MP.LEFT_ELBOW,     MP.LEFT_WRIST],
  ["rightArm",     MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW],
  ["rightForeArm",  MP.RIGHT_ELBOW,    MP.RIGHT_WRIST],
  ["leftUpLeg",    MP.LEFT_HIP,       MP.LEFT_KNEE],
  ["leftLeg",      MP.LEFT_KNEE,      MP.LEFT_ANKLE],
  ["rightUpLeg",   MP.RIGHT_HIP,      MP.RIGHT_KNEE],
  ["rightLeg",     MP.RIGHT_KNEE,     MP.RIGHT_ANKLE],
];

// ─── Coordinate conversion ───────────────────────────────────────────────────
// After our normalizer: x ≈ [-0.5, 0.5], y ≈ [-1, 0.3], z small depth.
// Babylon: x right, y up, z forward (left-handed).
const SCALE = 2.5;
const HIP_Y = 1.0;

function toV3([x, y, z]) {
  return new BABYLON.Vector3(
    -x * SCALE,
    -y * SCALE + HIP_Y,
     z * SCALE
  );
}

function mid(a, b) {
  return BABYLON.Vector3.Lerp(a, b, 0.5);
}

// ─── Quaternion: rotate direction A → direction B ────────────────────────────
function rotateAtoB(a, b) {
  a = a.clone().normalize();
  b = b.clone().normalize();
  const dot  = BABYLON.Vector3.Dot(a, b);
  if (dot > 0.99999) return BABYLON.Quaternion.Identity();
  if (dot < -0.99999) {
    // Pick an arbitrary perpendicular axis
    let perp = BABYLON.Vector3.Cross(BABYLON.Axis.X, a);
    if (perp.lengthSquared() < 1e-6) perp = BABYLON.Vector3.Cross(BABYLON.Axis.Y, a);
    return BABYLON.Quaternion.RotationAxis(perp.normalize(), Math.PI);
  }
  const axis  = BABYLON.Vector3.Cross(a, b).normalize();
  const angle = Math.acos(Math.max(-1, Math.min(1, dot)));
  return BABYLON.Quaternion.RotationAxis(axis, angle);
}

// ─── Fallback stick figure ───────────────────────────────────────────────────
function buildStickFigure(scene) {
  const color = new BABYLON.Color3(0.31, 0.56, 0.97);

  function makeCyl(name) {
    const m = BABYLON.MeshBuilder.CreateCylinder(name, { diameter: 0.06, height: 1, tessellation: 8 }, scene);
    const mat = new BABYLON.StandardMaterial(name + "_mat", scene);
    mat.diffuseColor = color; m.material = mat;
    return m;
  }
  function makeSphere(name, d = 0.1) {
    const m = BABYLON.MeshBuilder.CreateSphere(name, { diameter: d }, scene);
    const mat = new BABYLON.StandardMaterial(name + "_mat", scene);
    mat.diffuseColor = new BABYLON.Color3(1.0, 0.75, 0.15); m.material = mat;
    return m;
  }

  const head = makeSphere("sf_head", 0.25);
  const spine = makeCyl("sf_spine");
  const limbs = ["lu","ll","ru","rl","lul","lll","rul","rll"].map(n => makeCyl("sf_" + n));

  function updateCyl(mesh, a, b) {
    const dir = b.subtract(a); const len = dir.length();
    if (len < 1e-5) { mesh.isVisible = false; return; }
    mesh.isVisible = true;
    mesh.scaling.y = len;
    mesh.position = BABYLON.Vector3.Lerp(a, b, 0.5);
    mesh.rotationQuaternion = rotateAtoB(new BABYLON.Vector3(0,1,0), dir.normalize());
  }

  function applyStick(frame) {
    if (!frame?.pose) return;
    const p = frame.pose.map(toV3);
    head.position = p[MP.NOSE];
    updateCyl(spine, mid(p[MP.LEFT_HIP], p[MP.RIGHT_HIP]), mid(p[MP.LEFT_SHOULDER], p[MP.RIGHT_SHOULDER]));
    const segs = [
      [MP.LEFT_SHOULDER,MP.LEFT_ELBOW],[MP.LEFT_ELBOW,MP.LEFT_WRIST],
      [MP.RIGHT_SHOULDER,MP.RIGHT_ELBOW],[MP.RIGHT_ELBOW,MP.RIGHT_WRIST],
      [MP.LEFT_HIP,MP.LEFT_KNEE],[MP.LEFT_KNEE,MP.LEFT_ANKLE],
      [MP.RIGHT_HIP,MP.RIGHT_KNEE],[MP.RIGHT_KNEE,MP.RIGHT_ANKLE],
    ];
    segs.forEach(([a,b], i) => updateCyl(limbs[i], p[a], p[b]));
  }
  function dispose() { [head, spine, ...limbs].forEach(m => m.dispose()); }
  return { applyStick, dispose };
}

// ─── Main class ──────────────────────────────────────────────────────────────
export class AvatarAnimator {
  constructor(scene, avatarUrl) {
    this._scene      = scene;
    this._frames     = [];
    this._fps        = 30;
    this._elapsed    = 0;
    this._playing    = false;
    this._onComplete = null;
    this._skeleton   = null;
    this._mesh       = null;
    this._bones      = {};
    this._restDirs   = {};   // boneName → rest-pose world direction (parent→child)
    this._stick      = null;
    this._forceStickFigure = window.FORCE_STICK_DEBUG || false;  // For diagnostics
    this.onReady     = null;
    this.onError     = null;

    this._loadAvatar(avatarUrl);
  }

  async _loadAvatar(url) {
    try {
      const dir  = url.substring(0, url.lastIndexOf("/") + 1) || "./";
      const file = url.substring(url.lastIndexOf("/") + 1);
      const result = await BABYLON.SceneLoader.ImportMeshAsync("", dir, file, this._scene);

      const skinnedMesh = result.meshes.find(m => m.skeleton) ?? result.meshes[1];
      if (!skinnedMesh?.skeleton) throw new Error("No skeleton found.");

      this._mesh     = skinnedMesh;
      this._skeleton = skinnedMesh.skeleton;

      this._skeleton.bones.forEach(b => { this._bones[b.name] = b; });

      console.log("[AvatarAnimator] Bones:", Object.keys(this._bones).join(", "));

      // Auto-detect prefix (mixamorig: / mixamorig9: / etc.)
      const first = Object.keys(this._bones)[0] || "";
      const m = first.match(/^(mixamorig\d*:)/);
      const prefix = m ? m[1] : "";
      if (prefix) {
        console.log("[AvatarAnimator] Prefix:", prefix);
        for (const [k, v] of Object.entries(BONES)) {
          if (!v.startsWith(prefix)) BONES[k] = prefix + v;
        }
      }

      let resolved = 0;
      const missing = [];
      for (const [k, v] of Object.entries(BONES)) {
        const name = this._resolveBoneName(v);
        if (name) {
          BONES[k] = name;
          resolved += 1;
        } else {
          missing.push(v);
        }
      }
      console.log("[AvatarAnimator] Resolved bones:", resolved, "Missing:", missing.join(", "));

      if (this._forceStickFigure) {
        console.log("[AvatarAnimator] FORCE_STICK_DEBUG: ignoring rigged avatar, using stick figure");
        this._stick = buildStickFigure(this._scene);
        if (this.onReady) this.onReady();
        return;
      }

      // Capture rest-pose directions for FK chains.
      // In T-pose: upper arm points along ±X, forearm along ±X, legs along -Y.
      // We compute these from the actual bone positions in bind pose.
      this._skeleton.computeAbsoluteTransforms();
      this._captureRestDirections();

      if (this._stick) { this._stick.dispose(); this._stick = null; }
      if (this.onReady) this.onReady();
    } catch (err) {
      console.warn("[AvatarAnimator] Load failed:", err.message);
      this._stick = buildStickFigure(this._scene);
      if (this.onError) this.onError("Avatar not loaded — using stick figure fallback.");
    }
  }

  _resolveBoneName(target) {
    if (!target) return null;
    if (this._bones[target]) return target;

    const names = Object.keys(this._bones);
    const needle = target.toLowerCase();

    let found = names.find((n) => n.toLowerCase() === needle);
    if (found) return found;

    // Common rig separators/prefixes: mixamorig:Head, Armature_Hips, etc.
    const suffixes = [":" + needle, "_" + needle, "." + needle, "-" + needle];
    found = names.find((n) => {
      const low = n.toLowerCase();
      return suffixes.some((s) => low.endsWith(s));
    });
    if (found) return found;

    return names.find((n) => n.toLowerCase().endsWith(needle)) ?? null;
  }

  _captureRestDirections() {
    // For each FK chain, compute the world-space direction the bone points
    // in the rest/bind pose.  We do this by getting the bone's world position
    // and its child bone's world position.
    const boneWorldPos = {};
    this._skeleton.bones.forEach(b => {
      const wm = b.getWorldMatrix();
      boneWorldPos[b.name] = new BABYLON.Vector3(wm.m[12], wm.m[13], wm.m[14]);
    });

    for (const [key] of FK_CHAINS) {
      const boneName = BONES[key];
      const bone = this._bones[boneName];
      if (!bone) continue;

      // Pick the furthest valid child (twist children can be nearly zero-length).
      const parentPos = boneWorldPos[boneName];
      const children = bone.children.filter(c => c.name && boneWorldPos[c.name]);
      let bestDir = null;
      let bestLen = 0;

      for (const child of children) {
        const dir = boneWorldPos[child.name].subtract(parentPos);
        const len = dir.length();
        if (len > bestLen) {
          bestLen = len;
          bestDir = dir;
        }
      }

      if (bestDir && bestLen > 1e-5) {
        this._restDirs[boneName] = bestDir.normalize();
      } else {
        this._restDirs[boneName] = this._defaultRestDir(key);
      }
    }

    console.log("[AvatarAnimator] Rest directions captured for",
      Object.keys(this._restDirs).length, "bones");
  }

  _defaultRestDir(key) {
    if (key.includes("Arm") || key.includes("ForeArm")) {
      return key.startsWith("right")
        ? new BABYLON.Vector3(-1, 0, 0)
        : new BABYLON.Vector3(1, 0, 0);
    }
    if (key.includes("Leg") || key.includes("UpLeg")) {
      return new BABYLON.Vector3(0, -1, 0);
    }
    return new BABYLON.Vector3(0, 1, 0);
  }

  loadAnimation(data, onComplete = null) {
    this._frames   = data.frames ?? [];
    this._fps      = data.fps ?? 30;
    this._elapsed  = 0;
    this._playing  = this._frames.length > 0;
    this._onComplete = onComplete;
    this._loggedFirstFrame = false;
  }

  stop()   { this._playing = false; }
  replay() { this._elapsed = 0; this._playing = this._frames.length > 0; this._loggedFirstFrame = false; }

  get isPlaying() { return this._playing; }
  get progress()  { return this._frames.length ? Math.min(this._elapsed * this._fps / this._frames.length, 1) : 0; }

  update(delta, speed = 1) {
    if (!this._playing || !this._frames.length) return;
    this._elapsed += delta * speed;
    const idx = Math.floor(this._elapsed * this._fps);
    if (idx >= this._frames.length) {
      this._applyFrame(this._frames[this._frames.length - 1]);
      this._playing = false;
      if (this._onComplete) this._onComplete();
      return;
    }
    this._applyFrame(this._frames[idx]);
  }

  _applyFrame(frame) {
    if (!frame?.pose) return;

    if (!this._loggedFirstFrame) {
      this._loggedFirstFrame = true;
      console.log("[AvatarAnimator] First frame:", {
        poseLen: frame.pose.length,
        pose11: frame.pose[11], // left shoulder
        pose15: frame.pose[15], // left wrist
      });
    }

    if (this._stick) { this._stick.applyStick(frame); return; }
    if (!this._skeleton) return;

    // DIAGNOSTIC: test if bone rotations even affect the mesh
    if (window.AVATAR_TEST_BONE_ROTATION) {
      const testBone = this._bones[BONES.rightArm];
      if (testBone) {
        const angle = (this._elapsed * 3) % (Math.PI * 2); // varies 0-2π
        const testRot = BABYLON.Quaternion.FromEulerAngles(angle, 0, 0);
        testBone.setRotationQuaternion(testRot, BABYLON.Space.LOCAL);
        console.log("[AvatarAnimator] Test bone rotation applied: angle =", angle.toFixed(2));
      }
      return;
    }

    const pv = frame.pose.map(toV3);
    this._driveFK(pv);
    this._driveSpineHead(pv);
  }

  // ── FK: rotate each limb bone to match landmark direction ──────────────────
  _driveFK(pv) {
    this._skeleton.computeAbsoluteTransforms();

    for (const [key, fromIdx, toIdx] of FK_CHAINS) {
      const boneName = BONES[key];
      const bone = this._bones[boneName];
      if (!bone) continue;

      const restDir = this._restDirs[boneName];
      if (!restDir) continue;

      // Target direction from landmark positions
      const fromPos = pv[fromIdx];
      const toPos   = pv[toIdx];
      if (!fromPos || !toPos) continue;

      const targetDir = toPos.subtract(fromPos).normalize();
      if (targetDir.lengthSquared() < 1e-6) continue;

      // Compute the rotation that turns the rest direction into the target direction.
      // This is in WORLD space because both directions are world-space vectors.
      const worldRot = rotateAtoB(restDir, targetDir);

      bone.setRotationQuaternion(worldRot, BABYLON.Space.WORLD, this._mesh);

      // Keep absolute transforms fresh so child bones use updated parent orientation.
      this._skeleton.computeAbsoluteTransforms();
    }
  }

  // ── Spine + Head ───────────────────────────────────────────────────────────
  _driveSpineHead(pv) {
    const hipMid      = mid(pv[MP.LEFT_HIP], pv[MP.RIGHT_HIP]);
    const shoulderMid = mid(pv[MP.LEFT_SHOULDER], pv[MP.RIGHT_SHOULDER]);
    const spineDir    = shoulderMid.subtract(hipMid).normalize();
    const restUp      = new BABYLON.Vector3(0, 1, 0);

    // Spine2: lean/tilt
    const spine2 = this._bones[BONES.spine2];
    if (spine2) {
      const q = rotateAtoB(restUp, spineDir);
      spine2.setRotationQuaternion(q, BABYLON.Space.WORLD, this._mesh);
    }

    // Head: tilt toward nose
    const headBone = this._bones[BONES.head];
    if (headBone && pv[MP.NOSE]) {
      const neckPos = shoulderMid.add(new BABYLON.Vector3(0, 0.1, 0));
      const headDir = pv[MP.NOSE].subtract(neckPos).normalize();
      const q = rotateAtoB(restUp, headDir);
      headBone.setRotationQuaternion(q, BABYLON.Space.WORLD, this._mesh);
    }
  }
}
