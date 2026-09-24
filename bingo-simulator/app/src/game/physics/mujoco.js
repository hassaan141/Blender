// MuJoCo WASM boot and Bingo model loading.
//
// Framework-independent: no React, no Three.js. The runtime owns physics; the scene
// layer only reads state out of it.
//
// This targets the OFFICIAL @mujoco/mujoco bindings (DeepMind), whose API differs
// from the older community mujoco_wasm package: models are created with the static
// `MjModel.from_xml_path`, not `new Model(...)`, and enum values come through
// `mjtObj.mjOBJ_*.value`. Verified against the real module rather than assumed.

import {MUJOCO_URL, MJCF_SCENE, MODEL_DIR, JOINT_NAMES, DEFAULT_POSE} from "../constants.js";

let mujocoPromise = null;

/** Load the MuJoCo WASM module. Its .wasm sits beside the .js, so never bundle it. */
export function loadMujoco() {
  if (!mujocoPromise) {
    mujocoPromise = import(/* @vite-ignore */ MUJOCO_URL)
      .then((m) => (typeof m.default === "function" ? m.default() : m));
  }
  return mujocoPromise;
}

async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r.text();
}

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

/**
 * Compile the Bingo scene into MuJoCo's virtual filesystem and return the handles the
 * rest of the runtime needs. Every index is resolved BY NAME - Isaac orders DOFs
 * breadth-first and positional indexing scrambles the legs, a mistake this project
 * has already paid for once (see the track_v4 env's joint remap).
 */
export async function createBingoPhysics() {
  const mujoco = await loadMujoco();
  const OBJ = mujoco.mjtObj;
  const root = "/working";
  try { mujoco.FS.mkdir(root); } catch { /* already present */ }
  try { mujoco.FS.mkdir(`${root}/collision`); } catch { /* already present */ }

  const xml = await fetchText(MJCF_SCENE);
  const meshNames = [...new Set([...xml.matchAll(/file="([^"]+)"/g)].map((m) => m[1]))];
  await Promise.all(meshNames.map(async (name) => {
    mujoco.FS.writeFile(`${root}/collision/${name}`,
                        await fetchBytes(`${MODEL_DIR}/collision/${name}`));
  }));
  mujoco.FS.writeFile(`${root}/bingo_scene.xml`, xml);

  const model = mujoco.MjModel.from_xml_path(`${root}/bingo_scene.xml`);
  const data = new mujoco.MjData(model);

  const id = (type, name) => mujoco.mj_name2id(model, type.value, name);

  const jointId = {}, qposAdr = {}, dofAdr = {}, actId = {};
  for (const n of JOINT_NAMES) {
    const jid = id(OBJ.mjOBJ_JOINT, n);
    if (jid < 0) throw new Error(`MJCF has no joint named "${n}"`);
    jointId[n] = jid;
    qposAdr[n] = model.jnt_qposadr[jid];
    dofAdr[n] = model.jnt_dofadr[jid];
    const aid = id(OBJ.mjOBJ_ACTUATOR, n);
    if (aid < 0) throw new Error(`MJCF has no actuator named "${n}"`);
    actId[n] = aid;
  }
  const baseBody = id(OBJ.mjOBJ_BODY, "origin");
  const pawBody = {};
  for (const leg of ["fl", "fr", "bl", "br"]) {
    pawBody[leg] = id(OBJ.mjOBJ_BODY, `${leg}_knee`);
  }

  const sim = {
    mujoco, model, data, jointId, qposAdr, dofAdr, actId, baseBody, pawBody,
    step() { mujoco.mj_step(model, data); },
    forward() { mujoco.mj_forward(model, data); },
    /** Reset to the validated stance keyframe (key 0 = "stand"). */
    reset() {
      mujoco.mj_resetDataKeyframe(model, data, 0);
      for (let i = 0; i < JOINT_NAMES.length; i++) {
        data.ctrl[actId[JOINT_NAMES[i]]] = DEFAULT_POSE[i];
      }
      mujoco.mj_forward(model, data);
    },
    setTarget(name, value) { data.ctrl[actId[name]] = value; },
    getQ(name) { return data.qpos[qposAdr[name]]; },
    getQd(name) { return data.qvel[dofAdr[name]]; },
    getTarget(name) { return data.ctrl[actId[name]]; },
    getTorque(name) { return data.actuator_force[actId[name]]; },
    forceLimit(name) {
      const a = actId[name];
      return Math.abs(model.actuator_forcerange[a * 2 + 1]) || Infinity;
    },
    basePos() { return [data.qpos[0], data.qpos[1], data.qpos[2]]; },
    baseQuat() { return [data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]]; },
    bodyPos(name) { const b=id(OBJ.mjOBJ_BODY,name); return [data.xpos[3*b],data.xpos[3*b+1],data.xpos[3*b+2]]; },
    bodyQuat(name) { const b=id(OBJ.mjOBJ_BODY,name); return [data.xquat[4*b],data.xquat[4*b+1],data.xquat[4*b+2],data.xquat[4*b+3]]; },
    /**
     * Which paws are touching anything, as a {fl,fr,bl,br} boolean map.
     * `data.contact` is an Embind vector and goes stale after every step, so it is
     * re-read here each call rather than cached.
     */
    pawContacts() {
      const out = {fl: false, fr: false, bl: false, br: false};
      const vec = data.contact;
      const n = vec.size ? vec.size() : 0;
      for (let i = 0; i < n; i++) {
        const c = vec.get(i);
        for (const g of [c.geom1, c.geom2]) {
          const b = model.geom_bodyid[g];
          for (const leg of ["fl", "fr", "bl", "br"]) {
            if (pawBody[leg] === b) out[leg] = true;
          }
        }
      }
      if (vec.delete) vec.delete();   // Embind handles are not GC'd
      return out;
    },
    push(vx, vy, vz) {
      data.qvel[0] += vx; data.qvel[1] += vy; data.qvel[2] += vz;
    },
    dispose() { data.delete?.(); model.delete?.(); },
  };
  sim.reset();
  return sim;
}

/** Rotate a world vector into the base frame using the base quaternion (wxyz). */
export function worldToBase(q, v) {
  const [w, x, y, z] = q;
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [
    v[0] - w * tx + (y * tz - z * ty),
    v[1] - w * ty + (z * tx - x * tz),
    v[2] - w * tz + (x * ty - y * tx),
  ];
}

/** Tilt of the base from vertical, in degrees. */
export function tiltDeg(q) {
  const g = worldToBase(q, [0, 0, -1]);
  return (Math.acos(Math.max(-1, Math.min(1, -g[2]))) * 180) / Math.PI;
}
