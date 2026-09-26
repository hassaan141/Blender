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

// Ease a gesture in at this joint rate (rad/s) from whatever pose we were in.
const GESTURE_BLEND_RATE = 2.0;

import {
  CTRL_DT, DECIMATION, JOINT_NAMES, LEG_JOINTS, EXPR_JOINTS, NUM_JOINTS,
  DEFAULT_POSE, OBS_SIZE, ACTION_SIZE, STAND_BASE_HEIGHT, EXPR_OBS_TRAINING_MEAN,
} from "../constants.js";
import {createBingoPhysics, worldToBase, tiltDeg} from "../physics/mujoco.js";
import {fetchManifest, loadPolicy, PolicyRefused} from "../policies/loader.js";
import {ExpressionController} from "../controllers/expression.js";
import {SkillManager, State} from "../skills/manager.js";
import {loadCommandReferences, buildCommandObs, CommandState} from "./command_runtime.js";

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
    this.lastObservation = new Float32Array(OBS_SIZE);
    this.lastAction = new Float32Array(ACTION_SIZE);
    this.filteredResidual = new Float32Array(ACTION_SIZE);
    this.cmd = [0, 0]; this.cmdTarget = [0, 0]; this.phase = 0; this.refs = null;
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
      this.refs = await loadCommandReferences();
      this.commandState = new CommandState(this.refs);
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

  setCommand(vx, yaw) { this.cmdTarget[0] = Math.max(-.2, Math.min(.3, vx)); this.cmdTarget[1] = Math.max(-.6, Math.min(.6, yaw)); }

  reset() {
    this.sim.reset();
    this.skills.reset();
    this.lastAction.fill(0);
    this.filteredResidual.fill(0); this.phase = 0; this.cmd=[0,0]; this.cmdTarget=[0,0];
    this.expr.t = 0;
    if (this.refs) this.commandState = new CommandState(this.refs);
  }

  push(strength = 0.6) {
    const a = Math.random() * Math.PI * 2;
    this.sim.push(Math.cos(a) * strength, Math.sin(a) * strength, 0.05);
  }

  // ------------------------------------------------------------- observation
  /**
   * Build the 95-float observation in EXACTLY the training layout
   * (BASELINE_1/common.py). Order and content must match the env or the
   * policy is being fed noise that happens to be the right length.
   */
  buildObs() {
    const proprio = buildCommandObs(this.sim,this.cmd,this.phase,this.nextFeedforward(),this.filteredResidual).slice(0,67);
    const obs = this.commandState.observation(proprio);
    // Expressive joints enter the observation at their training values; see
    // EXPR_OBS_TRAINING_MEAN. The live values would saturate the policy's normalizer.
    for (let k = 0; k < 9; k++) {
      obs[12 + k] = EXPR_OBS_TRAINING_MEAN[k];
      obs[33 + k] = EXPR_OBS_TRAINING_MEAN[9 + k];
    }
    this.lastObservation.set(obs);
    return obs;
  }

  nextFeedforward() { return this.commandState ? this.commandState.feedforward() : DEFAULT_POSE.slice(0,12); }

  // ------------------------------------------------------------- control step
  async controlStep() {
    const sim = this.sim;

    // 1. expression: always runs, policy or not. The legs observe these joints.
    const exprTargets = this.expr.update(CTRL_DT);
    for (let k = 0; k < EXPR_JOINTS.length; k++) {
      sim.setTarget(EXPR_JOINTS[k], exprTargets[k]);
    }

    // 2. a running gesture. Which channels it drives depends on the clip:
    //    - expression-only clips (Yes/No/What) author ONLY head/tail/ears and leave
    //      the 12 leg channels at exactly zero. Applying those zeros commands the
    //      legs straight and topples the robot, so the legs are left to stand /
    //      locomotion below.
    //    - full-body clips drive all 21, eased in from the pose the gesture started
    //      from (they can begin 2.1 rad away from the stand pose).
    //    The root is never touched: the PD controllers track this under gravity and
    //    contact, exactly as Stage 4 does.
    // 2a. A TRACKED full-body skill (authored reference + learned leg residual,
    //     tools/skill_track): it drives all 21 joints and overrides expression.
    //     This is the Stage-5 style closed-loop correction that plain playback lacks.
    const tracked = this.skills.state === State.GESTURE ? this.skills.active?.tracker : null;
    let trackedStep = null;
    if (tracked) {
      if (tracked.k === null) tracked.start(sim, exprTargets);
      // First-order hold: the reference target is interpolated ref[k] -> ref[k+1] across
      // the physics substeps below (residual held), mirroring skill_env.py / the Isaac
      // Stage 4-5 envs exactly - a zero-order jump every 24 Hz control tick is what was
      // producing the shake this replaces.
      trackedStep = tracked.step(await tracked.policy.run(tracked.observe(sim)));
      this._gestureName = null;
    }

    const g = tracked ? null : this.skills.gestureFrame();
    if (g) {
      if (g.name !== this._gestureName) {      // new gesture: remember where we were
        this._gestureName = g.name;
        this._gestureFrom = JOINT_NAMES.map((n) => sim.getQ(n));
        // Ease in at a bounded joint rate. A clip can start 2.1 rad away from the
        // stand pose, and snapping there throws the robot; but a long blend would
        // swallow a short clip, so cap it at half the clip.
        // Only the channels the clip actually drives set the blend time. An
        // expression-only clip leaves the legs alone, so their (large) distance to
        // the clip's unused zero channels must not stretch the ease-in.
        let d = 0;
        for (let i = 0; i < NUM_JOINTS; i++) {
          const isLeg = LEG_IDX.includes(i);
          if (g.expressionOnly && isLeg) continue;
          d = Math.max(d, Math.abs(g.frame[i] - this._gestureFrom[i]));
        }
        this._gestureBlendS = Math.min(Math.max(d / GESTURE_BLEND_RATE, 0.25),
                                       Math.max(0.25, 0.5 * g.duration));
      }
      const blend = Math.max(0, Math.min(1, g.playhead / this._gestureBlendS));
      // Blend the expressive channels in too. Some clips (No) begin ALREADY at an
      // extreme - head_pitch -0.65 rad and tail_pitch 0.60 rad on frame 0 - so
      // applying frame 0 directly snaps the head and tail over and the impulse
      // flips the robot.
      for (let k = 0; k < EXPR_JOINTS.length; k++) {
        const gi = JOINT_NAMES.indexOf(EXPR_JOINTS[k]);
        const a = this._gestureFrom ? this._gestureFrom[gi] : g.frame[gi];
        sim.setTarget(EXPR_JOINTS[k], a + (g.frame[gi] - a) * blend);
      }
    } else {
      this._gestureName = null;
    }

    if (tracked) {
      /* legs already set by the tracked skill above */
    } else if (g && !g.expressionOnly) {
      const from = this._gestureFrom;
      const b = Math.max(0, Math.min(1, g.playhead / this._gestureBlendS));
      for (let k = 0; k < ACTION_SIZE; k++) {
        const gi = LEG_IDX[k];
        const a = from ? from[gi] : g.frame[gi];
        sim.setTarget(JOINT_NAMES[gi], a + (g.frame[gi] - a) * b);
      }
    } else if (this.policy &&
               ([State.WALK, State.STAND].includes(this.skills.state) ||
                (g && g.expressionOnly))) {
      // An expression-only gesture does not touch the legs, so locomotion keeps
      // running underneath it - the robot nods while the stand policy holds it up.
      // 3. legs from the policy. No policy -> the legs simply hold the stance; the
      // simulator never substitutes an animation for inference.
      const act = await this.policy.run(this.buildObs());
      this.lastAction.set(act);
      const {qTarget} = this.commandState.advance(act,this.cmdTarget);
      this.cmd = [...this.commandState.command];
      this.phase = this.commandState.phase;
      this.filteredResidual.set(this.commandState.filtered);
      for (let k = 0; k < ACTION_SIZE; k++) sim.setTarget(JOINT_NAMES[LEG_IDX[k]],qTarget[k]);
    } else {
      for (let k = 0; k < ACTION_SIZE; k++) {
        const gi = LEG_IDX[k];
        sim.setTarget(JOINT_NAMES[gi], DEFAULT_POSE[gi]);
      }
      this.lastAction.fill(0); this.filteredResidual.fill(0);
    }

    // 4. physics
    for (let s = 0; s < DECIMATION; s++) {
      if (trackedStep) {
        const {legs, expr} = tracked.targetsAt(trackedStep, (s + 1) / DECIMATION);
        for (let k = 0; k < 12; k++) sim.setTarget(JOINT_NAMES[k], legs[k]);
        for (let k = 0; k < EXPR_JOINTS.length; k++) sim.setTarget(EXPR_JOINTS[k], expr[k]);
      }
      sim.step();
      this._counters.phys++;
    }

    // 5. skill machine, from measured physical state
    if (tracked) tracked.check(sim);
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
      observation: Array.from(this.lastObservation),
    };
  }
}
