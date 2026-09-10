// Deterministic skill router. No learned router - the brief is explicit that the
// first simulator uses deterministic switching.
//
//   STAND <-> WALK
//     |        |
//   GESTURE   FALL -> RECOVER -> STAND
//
// Transitions use physical state with hysteresis (base height, tilt, contact count,
// dwell time), not raw key events. A key REQUESTS a transition; the machine decides
// whether it is currently legal. That is the difference between a state machine and
// a keyboard handler, and it is why a gesture cannot start mid-fall.

import {FALL_HEIGHT, FALL_TILT_DEG} from "../constants.js";

export const State = {
  STAND: "STAND",
  WALK: "WALK",
  GESTURE: "GESTURE",
  FALLEN: "FALLEN",
  RECOVER: "RECOVER",
};

export const SkillKind = {
  LOCOMOTION: "LOCOMOTION",
  RECOVERY: "RECOVERY",
  EXPRESSION: "EXPRESSION",
  GESTURE: "GESTURE",
  POSE: "POSE",
};

/**
 * A skill the runtime can execute.
 *
 * `ready` is deliberately explicit and defaults to false. A skill is exposed only if
 * something actually validated it - a Stage-3 animation existing is NOT sufficient,
 * because Stage-4 status is authoritative (MEMORY.md).
 */
export class BingoSkill {
  constructor({name, kind, ready = false, reason = "", motion = null, requiresPolicy = false}) {
    Object.assign(this, {name, kind, ready, reason, motion, requiresPolicy});
  }
}

// Fall detection must not chatter: a single frame past the threshold is noise, a
// sustained one is a fall. Matches the Stage-4/5 criterion plus a dwell.
const FALL_DWELL_S = 0.25;
const SETTLE_DWELL_S = 0.5;

export class SkillManager {
  constructor(skills = []) {
    this.skills = new Map(skills.map((s) => [s.name, s]));
    this.state = State.STAND;
    this.active = null;         // the running GESTURE/EXPRESSION skill
    this.playhead = 0;          // seconds into the active motion
    this._fallT = 0;
    this._settleT = 0;
    this.lastRefusal = null;
  }

  list(kind = null) {
    return [...this.skills.values()].filter((s) => !kind || s.kind === kind);
  }

  /** Ask to start a skill. Returns true if it actually started. */
  request(name) {
    const s = this.skills.get(name);
    if (!s) { this.lastRefusal = `no skill "${name}"`; return false; }
    if (!s.ready) {
      this.lastRefusal = `"${name}" is not validated for the simulator: ${s.reason}`;
      return false;
    }
    if (this.state === State.FALLEN || this.state === State.RECOVER) {
      this.lastRefusal = `cannot start "${name}" while ${this.state}`;
      return false;
    }
    if (s.requiresPolicy) {
      this.lastRefusal = `"${name}" needs a trained policy, which is not loaded`;
      return false;
    }
    this.active = s;
    this.playhead = 0;
    this.state = State.GESTURE;
    this.lastRefusal = null;
    return true;
  }

  /** Ask to walk. Only legal with a locomotion policy loaded. */
  requestWalk(hasPolicy) {
    if (this.state === State.FALLEN || this.state === State.RECOVER) return false;
    if (!hasPolicy) {
      this.lastRefusal = "no locomotion policy loaded - see policy_manifest.json";
      return false;
    }
    if (this.state !== State.GESTURE) this.state = State.WALK;
    return true;
  }

  requestStand() {
    if (this.state === State.WALK) this.state = State.STAND;
  }

  /**
   * Advance the machine.
   * @param dt seconds
   * @param phys {baseZ, tiltDeg, contacts} measured from MuJoCo this step
   */
  update(dt, phys) {
    const down = phys.baseZ < FALL_HEIGHT || phys.tiltDeg > FALL_TILT_DEG;

    if (this.state !== State.FALLEN && this.state !== State.RECOVER) {
      this._fallT = down ? this._fallT + dt : 0;
      if (this._fallT >= FALL_DWELL_S) {
        this.state = State.FALLEN;
        this.active = null;
        this._settleT = 0;
        return this.state;
      }
    }

    if (this.state === State.GESTURE && this.active) {
      this.playhead += dt;
      if (this.active.motion && this.playhead >= this.active.motion.duration) {
        this.active = null;
        this.state = State.STAND;
      }
    }

    if (this.state === State.FALLEN) {
      // Upright again and steady for a while (a shove that rights the robot counts).
      this._settleT = !down ? this._settleT + dt : 0;
      if (this._settleT >= SETTLE_DWELL_S && phys.contacts >= 3) {
        this.state = State.STAND;
      }
    }
    return this.state;
  }

  /** The current motion frame, if a motion skill is running. */
  motionFrame() {
    if (this.state !== State.GESTURE || !this.active?.motion) return null;
    return this.active.motion.sample(this.playhead);
  }

  reset() {
    this.state = State.STAND;
    this.active = null;
    this.playhead = 0;
    this._fallT = this._settleT = 0;
  }
}

/**
 * A Stage-3/4 robot-space motion, played as JOINT TARGETS.
 *
 * The floating root is never touched: the reference supplies joint angles and the PD
 * controllers track them under gravity and contact, exactly as Stage 4 does. If the
 * robot cannot hold the motion physically, it falls - which is the honest outcome and
 * the whole point of Stage-4 validation.
 */
export class RobotMotion {
  constructor({name, fps, frames}) {
    this.name = name;
    this.fps = fps;
    this.frames = frames;              // [T][21]
    this.duration = frames.length / fps;
  }

  static fromJson(j) {
    return new RobotMotion({name: j.name, fps: j.fps, frames: j.dof_positions});
  }

  /** Linear interpolation between reference frames, matching Stage 4's first-order hold. */
  sample(t) {
    const x = Math.max(0, Math.min(this.frames.length - 1, t * this.fps));
    const i = Math.floor(x);
    const j = Math.min(this.frames.length - 1, i + 1);
    const f = x - i;
    const a = this.frames[i], b = this.frames[j];
    const out = new Float32Array(a.length);
    for (let k = 0; k < a.length; k++) out[k] = a[k] + (b[k] - a[k]) * f;
    return out;
  }
}
