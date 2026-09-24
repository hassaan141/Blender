// Bingo's control contract. These are NOT Microduck's numbers.
//
// Microduck runs 50 Hz because that is its training rate. Bingo's Stage-4 and
// Stage-5 stack is 120 Hz physics with decimation 5 -> 24 Hz control, and copying
// the 50 rather than the principle would silently break every policy trained here.
// Source: stage5/bingo_stage5_env_cfg.py and rl/tools/track_v4_physics.py.

export const PHYSICS_HZ = 120;
export const TIMESTEP = 1 / PHYSICS_HZ;
export const DECIMATION = 5;
export const CONTROL_HZ = PHYSICS_HZ / DECIMATION; // 24
export const CTRL_DT = 1 / CONTROL_HZ;

// Canonical 21-DOF order, identical to v4_kinematics.DOF_ORDER and
// bake_conform.DOF_ORDER_21. Indices into MuJoCo are still resolved BY NAME at
// runtime - Isaac orders DOFs breadth-first and positional indexing scrambles the
// legs, a mistake this project has already made once (see track_v4 env).
export const JOINT_NAMES = [
  "fl_SY_J", "fl_SP_J", "fl_knee",
  "fr_SY_J", "fr_SP_J", "fr_knee",
  "bl_SY_J", "bl_SP_J", "bl_knee",
  "br_SY_J", "br_SP_J", "br_knee",
  "head_pitch_joint", "head_yaw", "head_roll",
  "tail_pitch", "tail_yaw",
  "l_ear_pitch", "l_ear_roll", "r_ear_pitch", "r_ear_roll",
];
export const NUM_JOINTS = 21;

// The policy acts on the 12 leg joints only. The 9 expressive joints are driven by
// the expression controller and are OBSERVED by the policy so it can compensate.
export const LEG_JOINTS = JOINT_NAMES.slice(0, 12);
export const EXPR_JOINTS = JOINT_NAMES.slice(12);
export const NUM_LEG = 12;
export const NUM_EXPR = 9;

// stage4/stand_test.py STAND_SOLVED. Verified against the real collision hulls:
// all four paws at exactly -0.180 m below the base, spread 0.00 mm.
// NOT the pose in BINGO_V4_CFG.init_state, which is the stale rev_3 pose and is not
// a valid v4 stance (docs/locomotion/TASK1_DESIGN.md).
export const DEFAULT_POSE = new Float32Array([
  0.0, +0.8100, +0.8932,
  0.0, -0.8109, -0.8938,
  0.0, +0.3932, +0.8913,
  0.0, +0.3936, -0.8913,
  0, 0, 0, 0, 0, 0, 0, 0, 0,
]);
export const STAND_BASE_HEIGHT = 0.182;

// Per-joint action scale, derived in docs/locomotion/TASK1_DESIGN.md so that
// |action| = 3 lands exactly on each joint's soft limit given its headroom from the
// stance. A single scalar cannot serve this robot: SY's range is +-0.42 rad against
// +-1.56 for every other leg joint.
export const ACTION_SCALE = new Float32Array([
  0.125, 0.195, 0.170,
  0.125, 0.195, 0.170,
  0.125, 0.195, 0.170,
  0.125, 0.195, 0.170,
]);

// 3 lin vel + 3 ang vel + 3 gravity + 3 command + 21 qpos + 21 qvel + 12 action
// heading-relative proprioception 67 + phase 2 + normalized command 2 +
// next feedforward leg target 12 + previous filtered residual 12.
export const OBS_SIZE = 95;
export const ACTION_SIZE = 12;

// Command limits, sized for a 0.18 m / 2.46 kg quadruped. Every authored Bingo clip
// walks at 0.06-0.14 m/s (rl/tools/analyze_style_motions.py), so these ask for
// meaningfully more than the animation while staying far under the Froude-3 ceiling
// of ~2.3 m/s.
export const VEL_FWD = 0.20, VEL_BACK = -0.20, VEL_LAT = 0.0, VEL_YAW = 0.4;
export const COMMAND_ALPHA = 0.1;
export const RESIDUAL_EMA_ALPHA = 0.3;
export const RESIDUAL_SCALE = 0.3;

// Actuator ceilings from bingo_v4.py, for the HUD's saturation readout.
export const EFFORT_LIMIT = {leg: 3.0, head: 6.0, tail: 6.0, ear: 1.0};

export const MODEL_DIR = "./robot";
export const MJCF_SCENE = `${MODEL_DIR}/bingo_scene.xml`;
export const KINEMATICS = `${MODEL_DIR}/kinematics.json`;
export const MANIFEST_URL = "./policy_manifest.json";
export const MOTION_DIR = "./motions";

// MuJoCo and ONNX Runtime are dynamically imported and NOT bundled: their .wasm
// sidecars resolve relative to the importing module's URL, so a bundler would look
// for them in the wrong place.
//
// They are VENDORED into public/vendor/ rather than loaded from a CDN (Microduck
// uses jsdelivr). Two reasons: the app then runs with no third-party network
// dependency at all, which matters for an offline demo and for locked-down networks;
// and the versions are pinned by package.json rather than by a URL, so `npm ci`
// reproduces them. tools/vendor_runtime.sh refreshes the copies.
// Set VITE_USE_CDN=1 to use jsdelivr instead.
//
// These MUST be absolute. A relative specifier passed to dynamic import() resolves
// against the IMPORTING MODULE's URL, not the page - so after a production build,
// "./vendor/mujoco/mujoco.js" became /bundle/vendor/mujoco/mujoco.js and 404'd,
// while every fetch() in the app kept working because fetch resolves against the
// document. Resolving against document.baseURI here makes both agree.
const CDN = import.meta.env?.VITE_USE_CDN === "1";
const local = (p) =>
  (typeof document !== "undefined" ? new URL(p, document.baseURI).href : p);

export const MUJOCO_URL = CDN
  ? "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.11.0/mujoco.js"
  : local("vendor/mujoco/mujoco.js");
export const ORT_URL = CDN
  ? "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.mjs"
  : local("vendor/ort/ort.min.mjs");
export const ORT_WASM_DIR = CDN
  ? "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/"
  : local("vendor/ort/");

// Fall detection, matching the Stage-4/Stage-5 criterion
// (stage5/bingo_stage5_env_cfg.py: root_z < 0.5*z_ref[0], tilt > 70 deg).
export const FALL_HEIGHT = 0.5 * STAND_BASE_HEIGHT;
export const FALL_TILT_DEG = 70;

// BASELINE_1 observation-normalizer means for the 9 expressive joints: positions
// (obs 12-20) then velocities (obs 33-41), EXPR_JOINTS order. The policy was trained
// with these joints static, so its normalizer std for them is ~0.0003-0.005 rad and the
// live expression (ears held at +-0.2 rad) saturates those inputs at +-5 sigma. The leg
// policy is therefore fed these training values instead; physics still runs the real
// expression. Browser gate: 2/9 -> 8/9 (docs/experiment_loop/sim2sim/python_browser_match.md).
export const EXPR_OBS_TRAINING_MEAN = new Float32Array([
  -0.006773754572658886, -0.014139362744507352, -0.006733530002451934, 0.0026720739463959127,
  0.00043322165482683985, -0.001620778714630172, -0.00037171963314017544, -0.0015291097243702407,
  9.869841833341386e-05,
  -0.008447833303575571, 0.0015250453908680658, 0.0003735199808641575, 0.0009366658652269389,
  -1.2980270498231318e-05, 0.0018321913605973713, -0.0020407356196855972, 0.001980557466711164,
  0.0021347867040716613,
]);
