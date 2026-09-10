// Task-2 Mode A: the deterministic expressive layer.
//
//   expression command -> head / tail / ear targets -> 9 PD-controlled joints
//
// This needs no trained network and works today. The leg policy OBSERVES these joints
// (all 21 are in the observation) so it can compensate for their dynamics, which is
// the whole reason the observation carries 21 and not 12.
//
// What the personality presets do NOT do
// --------------------------------------
// They do not change the gait, and the UI says so. Per the Task-2B audit already in
// this repo (docs/locomotion/STYLE_DATA_AUDIT.txt), the six authored clips do not
// contain separable personality gaits: only Deadpan and Laidback are physically
// plausible walking clips, and during walking the between-clip body-height difference
// (34.7 mm) is SMALLER than the within-clip standard deviation of either (28.9 and
// 44.3 mm). Claiming a "Cheeky walk" from that data would be inventing it.
//
// What IS measurably separable is expressive-channel activity - RMS joint rate over
// each clip, in rad/s, straight from that audit:
//
//     clip        tail   ear L   ear R   head
//     cheeky      1.80   1.89    1.99    1.05
//     timid       0.48   1.41    1.57    0.86
//     deadpan     0.64   0.70    0.87    0.69
//     laidback    0.35   1.03    1.15    0.40
//
// A 5x spread on the tail and 3x on the head, well outside measurement noise. The
// presets below are built from those measured rates - amplitude and frequency per
// channel - so the difference a viewer sees is the difference the data actually
// contains.

import {EXPR_JOINTS} from "../constants.js";

// Joint limits from the v4 URDF (rad). Targets are clamped to these; the expression
// layer must never command past a mechanical stop.
export const EXPR_LIMITS = {
  head_pitch_joint: [-0.65, 0.05],
  head_yaw: [-0.60, 0.60],
  head_roll: [-0.78, 0.78],
  tail_pitch: [-0.60, 0.60],
  tail_yaw: [-0.60, 0.60],
  l_ear_pitch: [-3.0, 3.0],
  l_ear_roll: [-1.50, 0.0],
  r_ear_pitch: [-3.0, 3.0],
  r_ear_roll: [0.0, 1.50],
};

// amp = peak amplitude (rad), hz = oscillation rate, bias = resting offset.
// Derived from the measured RMS rates above: a channel whose RMS rate is 5x another's
// gets roughly 5x the amp*hz product.
export const PERSONALITIES = {
  Neutral: {
    label: "Neutral", measured: false,
    head: {amp: 0.05, hz: 0.25, bias: 0.0},
    tail: {amp: 0.10, hz: 0.5, bias: 0.0},
    ear: {amp: 0.05, hz: 0.3, bias: -0.2},
  },
  Cheeky: {
    label: "Cheeky", measured: true, rms: {tail: 1.80, ear: 1.94, head: 1.05},
    head: {amp: 0.22, hz: 0.9, bias: -0.10},
    tail: {amp: 0.45, hz: 1.6, bias: 0.10},
    ear: {amp: 0.40, hz: 1.3, bias: -0.35},
  },
  Timid: {
    label: "Timid", measured: true, rms: {tail: 0.48, ear: 1.49, head: 0.86},
    head: {amp: 0.16, hz: 0.7, bias: -0.32},
    tail: {amp: 0.12, hz: 0.5, bias: -0.25},
    ear: {amp: 0.30, hz: 1.0, bias: -0.70},
  },
  DeadPan: {
    label: "DeadPan", measured: true, rms: {tail: 0.64, ear: 0.79, head: 0.69},
    head: {amp: 0.10, hz: 0.5, bias: -0.14},
    tail: {amp: 0.18, hz: 0.6, bias: 0.0},
    ear: {amp: 0.14, hz: 0.5, bias: -0.20},
  },
  Enthusiastic: {
    label: "Enthusiastic", measured: true, rms: {tail: 1.42, ear: 1.60, head: 1.04},
    head: {amp: 0.20, hz: 1.0, bias: -0.05},
    tail: {amp: 0.42, hz: 1.5, bias: 0.15},
    ear: {amp: 0.34, hz: 1.2, bias: -0.25},
  },
  Eccentric: {
    label: "Eccentric", measured: true, rms: {tail: 0.54, ear: 1.10, head: 0.45},
    head: {amp: 0.24, hz: 0.35, bias: -0.20},
    tail: {amp: 0.16, hz: 0.4, bias: -0.10},
    ear: {amp: 0.50, hz: 0.8, bias: -0.40},
  },
  Laidback: {
    label: "Laidback", measured: true, rms: {tail: 0.35, ear: 1.09, head: 0.40},
    head: {amp: 0.08, hz: 0.3, bias: -0.22},
    tail: {amp: 0.09, hz: 0.35, bias: -0.05},
    ear: {amp: 0.22, hz: 0.6, bias: -0.55},
  },
};

const clamp = (v, [lo, hi]) => Math.max(lo, Math.min(hi, v));

export class ExpressionController {
  constructor() {
    this.personality = "Neutral";
    this.t = 0;
    this.look = {yaw: 0, pitch: 0};   // operator head aim, radians
    this.targets = new Float32Array(EXPR_JOINTS.length);
    this.enabled = true;
  }

  setPersonality(name) {
    if (PERSONALITIES[name]) this.personality = name;
  }

  /** Operator head aim from mouse / right stick, in [-1, 1]. */
  setLook(yaw, pitch) {
    this.look.yaw = Math.max(-1, Math.min(1, yaw));
    this.look.pitch = Math.max(-1, Math.min(1, pitch));
  }

  /** Advance and return the 9 expressive joint targets, in EXPR_JOINTS order. */
  update(dt) {
    this.t += dt;
    const p = PERSONALITIES[this.personality] || PERSONALITIES.Neutral;
    const t = this.t;
    const osc = (c, phase = 0) => c.bias + c.amp * Math.sin(2 * Math.PI * c.hz * t + phase);

    // Operator aim is ADDED to the personality idle, so looking around never
    // suppresses the character and vice versa.
    const v = {
      head_pitch_joint: osc(p.head) + this.look.pitch * 0.30,
      head_yaw: osc(p.head, 1.1) * 0.6 + this.look.yaw * 0.55,
      head_roll: osc(p.head, 2.2) * 0.5,
      tail_pitch: osc(p.tail),
      tail_yaw: osc(p.tail, 1.6),
      // Ear roll ranges are one-sided and OPPOSITE per side in the URDF
      // (l_ear_roll [-1.5, 0], r_ear_roll [0, +1.5]), so a shared value must be
      // mirrored or one ear sits permanently at its stop.
      l_ear_pitch: osc(p.ear, 0.4),
      l_ear_roll: -Math.abs(osc(p.ear, 0.9)),
      r_ear_pitch: osc(p.ear, 0.4),
      r_ear_roll: Math.abs(osc(p.ear, 0.9)),
    };

    for (let i = 0; i < EXPR_JOINTS.length; i++) {
      const n = EXPR_JOINTS[i];
      this.targets[i] = this.enabled ? clamp(v[n], EXPR_LIMITS[n]) : 0;
    }
    return this.targets;
  }
}
