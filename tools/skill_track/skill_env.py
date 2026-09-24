"""MuJoCo skill-tracking env on the browser MJCF (120 Hz physics, 24 Hz control).

    legs (12):  target = ref_legs[k+1] + RESIDUAL_SCALE * clip(action, -1, 1)
    expr (9):   target = ref_expr[k+1]           (feed-forward, not learned)

Targets are held for the 5 physics substeps, exactly like the browser controlStep.
The root is never touched after reset. `k` is the reference frame the current state
corresponds to; the reference is anchored to the robot's pose at skill start
(`anchor` = x, y, yaw), which the browser does when the skill button is pressed.
Observation (82) — `observe()` is mirrored by the browser runtime; keep them in sync.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools/experiment_loop/sim2sim"))
from mujoco_env import neutral_expression, MJCF  # noqa: E402  (browser Neutral expression port)

JOINTS = ["fl_SY_J", "fl_SP_J", "fl_knee", "fr_SY_J", "fr_SP_J", "fr_knee",
          "bl_SY_J", "bl_SP_J", "bl_knee", "br_SY_J", "br_SP_J", "br_knee",
          "head_pitch_joint", "head_yaw", "head_roll", "tail_pitch", "tail_yaw",
          "l_ear_pitch", "l_ear_roll", "r_ear_pitch", "r_ear_roll"]
PAWS = ["fl_knee", "fr_knee", "bl_knee", "br_knee"]
TIP = np.array([0., 0., -.12])
RESIDUAL_SCALE = .25   # default; per-joint scales (12) are allowed and stored in the checkpoint config


def scale_from_config(cfg):
    s = (cfg or {}).get("residual_scale")
    return np.full(12, RESIDUAL_SCALE) if s is None else np.broadcast_to(np.asarray(s, np.float64), (12,)).copy()
OBS_DIM = 82
LOOKAHEAD = 4


def yaw_of(q): return math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
def qz(yaw): return np.array([math.cos(yaw/2), 0., 0., math.sin(yaw/2)])
def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])
def qconj(q): return np.array([q[0], -q[1], -q[2], -q[3]])
def qrot(q, v):
    t = 2*np.cross(q[1:], v); return v + q[0]*t + np.cross(q[1:], t)


class Reference:
    def __init__(self, path):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        for k in ("legs", "expr", "root_pos", "root_quat", "tips", "contacts"): setattr(self, k, z[k])
        self.n = len(self.legs); self.tin = self.meta["tin"]


class SkillEnv:
    def __init__(self, ref_path, seed=0, stand_prob=.3, focus=None, residual_scale=None, push=0.):
        self.push = push  # training only: per-step probability of a random root velocity kick (sigma 0.1 m/s)
        self.rscale = scale_from_config({"residual_scale": residual_scale})
        self.focus = focus  # (lo, hi, prob): extra reference-state starts in a known-weak frame range
        self.stand_prob = stand_prob  # share of episodes starting from standing (the browser case)
        self.rng = np.random.default_rng(seed)
        self.m = mujoco.MjModel.from_xml_path(str(MJCF)); self.d = mujoco.MjData(self.m)
        assert abs(self.m.opt.timestep - 1/120) < 1e-9
        J = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.qa = np.array([self.m.jnt_qposadr[J(n)] for n in JOINTS]); self.da = np.array([self.m.jnt_dofadr[J(n)] for n in JOINTS])
        self.aid = np.array([mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS])
        self.paw = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in PAWS]
        self.stand = self.m.key_qpos[0].copy()
        self.ref = Reference(ref_path)
        self.prev_action = np.zeros(12); self.k = 0

    # ---- reference in world frame (anchored) -------------------------------------
    def rk(self, k): return min(max(k, 0), self.ref.n - 1)
    def ref_pos(self, k):
        p = self.ref.root_pos[self.rk(k)]; return np.array([*self.anchor[:2], 0.]) + qrot(self.aq, p)
    def ref_quat(self, k): return qmul(self.aq, self.ref.root_quat[self.rk(k)])
    def ref_expr(self, k):
        k = self.rk(k)
        if k < self.ref.tin:  # entering: blend from the live expression at skill start
            w = k / self.ref.tin; return (1-w)*self.expr0 + w*self.ref.expr[self.ref.tin]
        return self.ref.expr[k]

    # ---- episode --------------------------------------------------------------------
    def reset(self, frame=None, stand_start=None, noise=True, anchor=None):
        m, d, rng = self.m, self.d, self.rng
        mujoco.mj_resetDataKeyframe(m, d, 0)
        self.anchor = np.array(anchor if anchor is not None else
                               [rng.uniform(-.5, .5), rng.uniform(-.5, .5), rng.uniform(-math.pi, math.pi)] if noise else [0., 0., 0.])
        self.aq = qz(self.anchor[2])
        self.expr0 = neutral_expression(rng.uniform(0, 20)) if noise else neutral_expression(0.)
        if stand_start is None: stand_start = rng.random() < self.stand_prob
        if frame is None and not stand_start and self.focus and rng.random() < self.focus[2]:
            frame = int(rng.integers(int(self.focus[0]), int(self.focus[1]) + 1))
        k = 0 if stand_start else (frame if frame is not None else int(rng.integers(0, self.ref.n - 2)))
        self.k = k
        if stand_start:  # the browser case: robot standing with the live expression
            d.qpos[:2] = self.anchor[:2]; d.qpos[3:7] = self.aq
            d.qpos[self.qa[12:]] = self.expr0
        else:            # reference-state initialisation, zero velocity
            d.qpos[:3] = self.ref_pos(k); d.qpos[3:7] = self.ref_quat(k)
            d.qpos[self.qa[:12]] = self.ref.legs[k]; d.qpos[self.qa[12:]] = self.ref_expr(k)
        if noise: d.qpos[self.qa[:12]] += rng.normal(0, .03, 12)
        mujoco.mj_forward(m, d)
        pen = min([d.contact[i].dist for i in range(d.ncon)] + [0.])
        if pen < 0: d.qpos[2] += -pen + 1e-3; mujoco.mj_forward(m, d)
        self.prev_action = np.zeros(12)
        return self.observe()

    def paw_tips(self):
        return np.array([self.d.xpos[b] + qrot(self.d.xquat[b], TIP) for b in self.paw])

    def observe(self):
        d = self.d; q = d.qpos[3:7]; qi = qconj(q)
        yaw = yaw_of(q); hq = qz(-yaw)
        k1, k4 = self.k + 1, self.k + LOOKAHEAD
        rq1 = self.ref_quat(k1)
        perr = qrot(hq, self.ref_pos(k1) - d.qpos[:3])
        dyaw = yaw_of(rq1) - yaw
        s = self.k / (self.ref.n - 1)
        return np.concatenate([
            qrot(qi, np.array([0., 0., -1.])), d.qvel[3:6], qrot(qi, d.qvel[:3]), [d.qpos[2]],
            d.qpos[self.qa[:12]], d.qvel[self.da[:12]], self.prev_action,
            self.ref.legs[self.rk(k1)], self.ref.legs[self.rk(k4)],
            [self.ref_pos(k1)[2]], qrot(qconj(rq1), np.array([0., 0., -1.])),
            perr, [math.sin(dyaw), math.cos(dyaw)], [s, math.sin(2*math.pi*s), math.cos(2*math.pi*s)],
        ]).astype(np.float32)

    def step(self, action, residual=True):
        m, d = self.m, self.d
        a = np.clip(np.asarray(action, np.float64), -1, 1) if residual else np.zeros(12)
        kt = self.rk(self.k + 1)
        d.ctrl[self.aid[:12]] = self.ref.legs[kt] + self.rscale * a
        d.ctrl[self.aid[12:]] = self.ref_expr(kt)
        if self.push and self.rng.random() < self.push: d.qvel[:2] += self.rng.normal(0, .1, 2)
        for _ in range(5): mujoco.mj_step(m, d)
        self.k = kt
        # ---- reward vs reference frame k ----
        q = d.qpos[3:7]; p = d.qpos[:3]
        rq, rp = self.ref_quat(kt), self.ref_pos(kt)
        ang = 2*math.acos(min(1., abs(float(np.dot(q, rq)))))
        dz = p[2] - rp[2]; dxy = float(np.linalg.norm(p[:2] - rp[:2]))
        ej = float(np.mean((d.qpos[self.qa[:12]] - self.ref.legs[kt])**2))
        hq, rhq = qz(-yaw_of(q)), qz(-yaw_of(rq))
        tips = self.paw_tips()
        rtips = np.array([np.array([*self.anchor[:2], 0.]) + qrot(self.aq, t) for t in self.ref.tips[kt]])
        et = float(np.mean(np.sum((np.array([qrot(hq, t - p) for t in tips]) - np.array([qrot(rhq, t - rp) for t in rtips]))**2, 1)))
        contact = np.array([self._in_contact(b) for b in self.paw], np.float64)
        cmatch = float(np.mean(contact == (self.ref.contacts[kt] > .5)))
        torque = d.actuator_force[self.aid[:12]]
        r = (.30*math.exp(-ej/.04) + .20*math.exp(-(ang/.25)**2) + .15*math.exp(-(dz/.03)**2)
             + .10*math.exp(-(dxy/.10)**2) + .15*math.exp(-et/.0015) + .05*cmatch + .05
             - .03*float(np.mean(a**2)) - .02*float(np.mean((a - self.prev_action)**2)) - .01*float(np.mean((torque/3.)**2)))
        self.prev_action = a
        fell = ang > .9 or p[2] < .04 or abs(dz) > .08 or dxy > .3 or not np.isfinite(d.qpos).all()
        done = fell or self.k >= self.ref.n - 1
        info = dict(fell=fell, ang=ang, dz=dz, dxy=dxy, joint_rms=math.sqrt(ej), tip_rms=math.sqrt(et),
                    cmatch=cmatch, torque=torque.copy(), residual=self.rscale*a, contact=contact, tips=tips)
        return self.observe(), r, done, info

    def _in_contact(self, body):
        d = self.d
        for i in range(d.ncon):
            c = d.contact[i]; b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if (b1 == body and b2 == 0) or (b2 == body and b1 == 0): return 1.
        return 0.
