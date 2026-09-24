// Tracked full-body skills: the authored motion is the REFERENCE, a learned residual
// policy supplies the balance corrections on the 12 legs.
//
//   legs (12):  target = ref.legs[k+1] + residual_scale[i] * clip(action, -1, 1)
//   expr (9):   target = reference feed-forward (entering blend starts from the live expression)
//
// This is a line-by-line mirror of tools/skill_track/skill_env.py (observe / step /
// termination). The two MUST stay in sync; the parity test compares them numerically.
// The root is never touched: the reference is only re-anchored to where the robot is
// standing when the skill starts.

import {JOINT_NAMES} from "../constants.js";
import {loadOrt} from "../policies/loader.js";

const LOOKAHEAD = 4;
const yawOf = (q) => Math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]*q[2]+q[3]*q[3]));
const qz = (yaw) => [Math.cos(yaw/2), 0, 0, Math.sin(yaw/2)];
const qconj = (q) => [q[0], -q[1], -q[2], -q[3]];
const qmul = (a, b) => [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
  a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]];
const qrot = (q, v) => {
  const t = [2*(q[2]*v[2]-q[3]*v[1]), 2*(q[3]*v[0]-q[1]*v[2]), 2*(q[1]*v[1]-q[2]*v[0])];
  return [v[0]+q[0]*t[0]+q[2]*t[2]-q[3]*t[1], v[1]+q[0]*t[1]+q[3]*t[0]-q[1]*t[2], v[2]+q[0]*t[2]+q[1]*t[1]-q[2]*t[0]];
};
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

export async function loadTrackedSkill(entry) {
  const r = await fetch(entry.reference);
  if (!r.ok) throw new Error(`skill reference ${entry.reference}: ${r.status}`);
  const ref = await r.json();
  if (ref.joint_order.some((n, i) => n !== JOINT_NAMES[i])) throw new Error(`${entry.name}: reference joint order differs`);
  const ort = await loadOrt();
  const session = await ort.InferenceSession.create(entry.policy);
  const out = session.outputNames.includes("actions") ? "actions" : session.outputNames[0];
  return new SkillTracker(ref, entry, {
    async run(obs) {
      const o = await session.run({[session.inputNames[0]]: new ort.Tensor("float32", obs, [1, obs.length])});
      return o[out].data;
    },
  });
}

export class SkillTracker {
  constructor(ref, entry, policy) {
    this.ref = ref; this.n = ref.frames; this.tin = ref.tin; this.policy = policy;
    this.obsDim = entry.obs_dim;
    const s = entry.residual_scale;   // one value or 12 per-joint values (rad per unit action)
    this.scale = Array.isArray(s) ? s : new Array(12).fill(s);
    this.k = null;              // null = not started; set by start()
  }

  rk(k) { return Math.min(Math.max(k, 0), this.n - 1); }
  refPos(k) { const p = qrot(this.aq, this.ref.root_pos[this.rk(k)]); return [this.anchor[0]+p[0], this.anchor[1]+p[1], p[2]]; }
  refQuat(k) { return qmul(this.aq, this.ref.root_quat[this.rk(k)]); }
  refExpr(k) {
    k = this.rk(k);
    if (k < this.tin) { const w = k / this.tin, e = this.ref.expr[this.tin]; return this.expr0.map((v, i) => (1-w)*v + w*e[i]); }
    return this.ref.expr[k];
  }

  /** Anchor the reference at the robot's current pose; expr0 = live expression targets. */
  start(sim, expr0) {
    const d = sim.data, q = [d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]];
    this.anchor = [d.qpos[0], d.qpos[1], yawOf(q)]; this.aq = qz(this.anchor[2]);
    this.expr0 = Array.from(expr0); this.k = 0; this.prevAction = new Array(12).fill(0);
    this.done = false; this.fallen = false;
  }

  observe(sim) {
    const d = sim.data, q = [d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]], qi = qconj(q);
    const yaw = yawOf(q), hq = qz(-yaw), k1 = this.k + 1, k4 = this.k + LOOKAHEAD;
    const rq1 = this.refQuat(k1), rp1 = this.refPos(k1);
    const perr = qrot(hq, [rp1[0]-d.qpos[0], rp1[1]-d.qpos[1], rp1[2]-d.qpos[2]]);
    const dyaw = yawOf(rq1) - yaw, s = this.k / (this.n - 1);
    const o = [
      ...qrot(qi, [0, 0, -1]), d.qvel[3], d.qvel[4], d.qvel[5], ...qrot(qi, [d.qvel[0], d.qvel[1], d.qvel[2]]), d.qpos[2],
      ...JOINT_NAMES.slice(0, 12).map((n) => d.qpos[sim.qposAdr[n]]),
      ...JOINT_NAMES.slice(0, 12).map((n) => d.qvel[sim.dofAdr[n]]),
      ...this.prevAction, ...this.ref.legs[this.rk(k1)], ...this.ref.legs[this.rk(k4)],
      rp1[2], ...qrot(qconj(rq1), [0, 0, -1]), ...perr, Math.sin(dyaw), Math.cos(dyaw),
      s, Math.sin(2*Math.PI*s), Math.cos(2*Math.PI*s),
    ];
    if (o.length !== this.obsDim) throw new Error(`skill observation ${o.length} != ${this.obsDim}`);
    return Float32Array.from(o);
  }

  /** Targets for the next control step; advances the reference frame. */
  step(action) {
    const a = Array.from(action, (v) => clamp(v, -1, 1)), kt = this.rk(this.k + 1);
    const legs = this.ref.legs[kt].map((v, i) => v + this.scale[i] * a[i]);
    const expr = this.refExpr(kt);
    this.prevAction = a; this.k = kt;
    return {legs, expr};
  }

  /** Called after physics: same termination as skill_env.py. */
  check(sim) {
    const d = sim.data, q = [d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]];
    const rq = this.refQuat(this.k), rp = this.refPos(this.k);
    const dot = Math.abs(q[0]*rq[0]+q[1]*rq[1]+q[2]*rq[2]+q[3]*rq[3]);
    const ang = 2*Math.acos(Math.min(1, dot)), dz = d.qpos[2] - rp[2], dxy = Math.hypot(d.qpos[0]-rp[0], d.qpos[1]-rp[1]);
    this.fallen = ang > .9 || d.qpos[2] < .04 || Math.abs(dz) > .08 || dxy > .3;
    this.done = this.k >= this.n - 1;
    return {ang, dz, dxy};
  }
}
