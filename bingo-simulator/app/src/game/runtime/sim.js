// The Bingo runtime: physics, policy, expression and skills on a fixed-step clock.
//
//   command -> policy (12 leg actions) -> position targets -> MuJoCo -> robot state
//   expression command -> 9 expressive targets ------------^
//
// Framework-independent. React reads a snapshot; it never drives the loop.
//
// Timing (constants.js): 120 Hz physics, decimation 5, 24 Hz policy. The accumulator
// is wall-clock and INDEPENDENT of rendering - the brief and Microduck both insist on
// this, and Microduck's loop carries the same "fell behind: don't spiral" guard,
// without which a slow tab death-spirals trying to catch up.

import {
  CTRL_DT, DECIMATION, JOINT_NAMES, LEG_JOINTS, EXPR_JOINTS, NUM_JOINTS,
  DEFAULT_POSE, OBS_SIZE, ACTION_SIZE, STAND_BASE_HEIGHT,
} from "../constants.js";
import {createBingoPhysics, worldToBase, tiltDeg} from "../physics/mujoco.js";
import {fetchManifest, loadPolicy, PolicyRefused} from "../policies/loader.js";
import {ExpressionController} from "../controllers/expression.js";
import {SkillManager, State} from "../skills/manager.js";

const LEG_IDX = LEG_JOINTS.map((n) => JOINT_NAMES.indexOf(n));
const EXPR_IDX = EXPR_JOINTS.map((n) => JOINT_NAMES.indexOf(n));

export class BingoRuntime {
  constructor() {
    this.sim = null;
    this.policy = null;
    this.manifest = null;
    this.policyError = null;
    this.expr = new ExpressionController();
    this.skills = new SkillManager();
    this.obs = new Float32Array(OBS_SIZE);
    this.lastAction = new Float32Array(ACTION_SIZE);
    this.cmd = [0, 0, 0];
    this.running = false;
    this.ctrlHz = 0;
    this.physHz = 0;
    this._counters = {ctrl: 0, phys: 0, t0: 0};
    this.stats = {};
  }

  async init({skills = []} = {}) {
    this.sim = await createBingoPhysics();
    for (const s of skills) this.skills.skills.set(s.name, s);

    try {
      this.manifest = await fetchManifest();
      this.policy = await loadPolicy(this.manifest);
      if (!this.policy) {
        this.policyError = this.manifest.status_detail
          || "the manifest declares no policies";
      }
    } catch (e) {
      // A refused policy is reported, never bypassed.
      this.policyError = e instanceof PolicyRefused
        ? `POLICY REFUSED\n${e.message}` : `manifest/policy load failed: ${e.message}`;
      this.policy = null;
    }
    return this;
  }

  get hasPolicy() { return this.policy !== null; }

  setCommand(vx, vy, wz) { this.cmd[0] = vx; this.cmd[1] = vy; this.cmd[2] = wz; }

  reset() {
    this.sim.reset();
    this.skills.reset();
    this.lastAction.fill(0);
    this.expr.t = 0;
  }

  push(strength = 0.6) {
    const a = Math.random() * Math.PI * 2;
    this.sim.push(Math.cos(a) * strength, Math.sin(a) * strength, 0.05);
  }

  // ------------------------------------------------------------- observation
  /**
   * Build the 66-float observation in EXACTLY the training layout
   * (docs/locomotion/TASK1_DESIGN.md). Order and content must match the env or the
   * policy is being fed noise that happens to be the right length.
   */
  buildObs() {
    const {data, qposAdr, dofAdr} = this.sim;
    const o = this.obs;
    const q = this.sim.baseQuat();
    let i = 0;

    // base linear + angular velocity, in the BODY frame (qvel[0..5] is world for a
    // free joint, so it has to be rotated)
    const lin = worldToBase(q, [data.qvel[0], data.qvel[1], data.qvel[2]]);
    const ang = worldToBase(q, [data.qvel[3], data.qvel[4], data.qvel[5]]);
    o[i++] = lin[0]; o[i++] = lin[1]; o[i++] = lin[2];
    o[i++] = ang[0]; o[i++] = ang[1]; o[i++] = ang[2];

    const g = worldToBase(q, [0, 0, -1]);
    o[i++] = g[0]; o[i++] = g[1]; o[i++] = g[2];

    o[i++] = this.cmd[0]; o[i++] = this.cmd[1]; o[i++] = this.cmd[2];

    // ALL 21 joints, relative to the stance pose
    for (let j = 0; j < NUM_JOINTS; j++) {
      o[i++] = data.qpos[qposAdr[JOINT_NAMES[j]]] - DEFAULT_POSE[j];
    }
    for (let j = 0; j < NUM_JOINTS; j++) {
      o[i++] = data.qvel[dofAdr[JOINT_NAMES[j]]];
    }
    // previous LEG actions only
    for (let j = 0; j < ACTION_SIZE; j++) o[i++] = this.lastAction[j];

    if (i !== OBS_SIZE) throw new Error(`observation built ${i} floats, expected ${OBS_SIZE}`);
    return o;
  }

  // ------------------------------------------------------------- control step
  async controlStep() {
    const sim = this.sim;

    // 1. expression: always runs, policy or not. The legs observe these joints.
    const exprTargets = this.expr.update(CTRL_DT);
    for (let k = 0; k < EXPR_JOINTS.length; k++) {
      sim.setTarget(EXPR_JOINTS[k], exprTargets[k]);
    }

    // 2. a running gesture overrides ALL 21 joints with its validated reference.
    //    The root is never touched - the PD controllers track it under gravity and
    //    contact, exactly as Stage 4 does.
    const frame = this.skills.motionFrame();
    if (frame) {
      for (let k = 0; k < NUM_JOINTS; k++) sim.setTarget(JOINT_NAMES[k], frame[k]);
    } else if (this.policy && this.skills.state === State.WALK) {
      // 3. legs from the policy. No policy -> the legs simply hold the stance; the
      //    simulator never substitutes an animation for inference.
      const act = await this.policy.run(this.buildObs());
      this.lastAction.set(act);
      const scale = this.policy.actionScale;
      for (let k = 0; k < ACTION_SIZE; k++) {
        const gi = LEG_IDX[k];
        sim.setTarget(JOINT_NAMES[gi], DEFAULT_POSE[gi] + act[k] * scale[k]);
      }
    } else {
      for (let k = 0; k < ACTION_SIZE; k++) {
        const gi = LEG_IDX[k];
        sim.setTarget(JOINT_NAMES[gi], DEFAULT_POSE[gi]);
      }
      this.lastAction.fill(0);
    }

    // 4. physics
    for (let s = 0; s < DECIMATION; s++) {
      sim.step();
      this._counters.phys++;
    }

    // 5. skill machine, from measured physical state
    const m = this.measure();
    this.skills.update(CTRL_DT, m);
    this.stats = m;
    this._counters.ctrl++;
  }

  measure() {
    const sim = this.sim;
    const q = sim.baseQuat();
    const lin = worldToBase(q, [sim.data.qvel[0], sim.data.qvel[1], sim.data.qvel[2]]);
    const ang = worldToBase(q, [sim.data.qvel[3], sim.data.qvel[4], sim.data.qvel[5]]);
    const g = worldToBase(q, [0, 0, -1]);

    const contacts = sim.pawContacts();
    const contactCount = Object.values(contacts).filter(Boolean).length;

    let maxTorque = 0, satCount = 0;
    for (const name of JOINT_NAMES) {
      const f = Math.abs(sim.getTorque(name));
      if (f > maxTorque) maxTorque = f;
      if (f >= 0.99 * sim.forceLimit(name)) satCount++;
    }

    return {
      baseZ: sim.data.qpos[2],
      basePos: sim.basePos(),
      tiltDeg: tiltDeg(q),
      roll: Math.atan2(g[1], -g[2]) * 180 / Math.PI,
      pitch: Math.atan2(-g[0], -g[2]) * 180 / Math.PI,
      vx: lin[0], vy: lin[1], yawRate: ang[2],
      contacts,
      contactCount,
      maxTorque,
      saturated: satCount,
      state: this.skills.state,
    };
  }

  // ------------------------------------------------------------- the loop
  start() {
    if (this.running) return;
    this.running = true;
    this._counters.t0 = performance.now();
    let next = performance.now();

    const loop = async () => {
      while (this.running) {
        try {
          await this.controlStep();
        } catch (e) {
          this.running = false;
          this.policyError = `runtime stopped: ${e.message}`;
          throw e;
        }
        const now = performance.now();
        const el = now - this._counters.t0;
        if (el > 500) {
          this.ctrlHz = (this._counters.ctrl * 1000) / el;
          this.physHz = (this._counters.phys * 1000) / el;
          this._counters.ctrl = this._counters.phys = 0;
          this._counters.t0 = now;
        }
        next += CTRL_DT * 1000;
        const wait = next - performance.now();
        if (wait > 0) await new Promise((r) => setTimeout(r, wait));
        else next = performance.now();   // fell behind: don't spiral
      }
    };
    loop();
  }

  stop() { this.running = false; }

  /** Everything the UI needs, as plain data. */
  snapshot() {
    const jointPos = {}, jointTarget = {};
    for (const n of JOINT_NAMES) {
      jointPos[n] = this.sim.getQ(n);
      jointTarget[n] = this.sim.getTarget(n);
    }
    return {
      ...this.stats,
      cmd: [...this.cmd],
      ctrlHz: this.ctrlHz,
      physHz: this.physHz,
      hasPolicy: this.hasPolicy,
      policyName: this.policy?.name ?? null,
      policyError: this.policyError,
      personality: this.expr.personality,
      skillState: this.skills.state,
      activeSkill: this.skills.active?.name ?? null,
      lastRefusal: this.skills.lastRefusal,
      jointPos, jointTarget,
    };
  }
}
