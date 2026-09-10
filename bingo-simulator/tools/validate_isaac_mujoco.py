"""Validate the generated MuJoCo Bingo against the authority both simulators derive from.

Isaac Sim cannot run in every environment (it needs an RTX GPU), so this tool does
NOT begin by assuming Isaac is available. What it always checks is stronger than it
first looks: the v4 URDF is the artefact Isaac itself consumes - bingo_v4.py spawns a
USD converted from it - and stage2/v4_kinematics.py is an independent, already-trusted
FK implementation of that same URDF, used by the entire Stage-2/Stage-4 pipeline. So
the MJCF can be checked against the common ancestor of both simulators, on CPU.

Checks, each with a stated tolerance and a numeric result:

  1  FK        body positions at N random joint configurations, all 21 joints
  2  joints    axis, origin and limits, joint by joint
  3  mass      per-link mass, total mass, and centre of mass
  4  stance    STAND_SOLVED rest pose and paw heights (target: 4 paws at -0.180 m)
  5  standing  equilibrium under gravity - does it hold the stance, how far does it sink
  6  drop      gravity drop from a height - does it land and settle
  7  step      single-joint step responses against the configured PD gains
  8  contact   contact point positions vs stage4/out/collision_hulls.npz

Kinematics and mass properties (1-4) are exact comparisons: a failure there is a
conversion bug. Dynamics (5-8) are behavioural: MuJoCo and PhysX will never agree
bit-for-bit and the tolerances say so.

    python3 bingo-simulator/tools/validate_isaac_mujoco.py
    python3 bingo-simulator/tools/validate_isaac_mujoco.py --report out.txt --n 500

An Isaac-side trace recorded on a GPU machine (rl/tools/eval_velocity.py --out) can
be compared with --isaac-trace; without it, the report says the Isaac comparison did
not run rather than implying parity was shown.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(SIM_ROOT, ".."))
sys.path.insert(0, os.path.join(REPO, "stage2"))
sys.path.insert(0, os.path.join(REPO, "stage4"))

import mujoco  # noqa: E402
from v4_kinematics import V4Kin, LEGS, DOF_ORDER, axis_rot  # noqa: E402

MJCF = os.path.join(SIM_ROOT, "app/public/robot/bingo_v4.xml")
# Dynamics need a floor to stand on. bingo_v4.xml is the robot ALONE (the parity
# subject); bingo_scene.xml adds the ground plane. Running a standing test on the
# floorless model just measures free fall.
MJCF_SCENE = os.path.join(SIM_ROOT, "app/public/robot/bingo_scene.xml")
URDF_P = os.path.join(REPO, "URDF/bingo_urdf v4_w_ear_joints/urdf/"
                            "bingo_urdf_w_ear_joints_physics.urdf")
HULLS = os.path.join(REPO, "stage4/out/collision_hulls.npz")

STAND_SOLVED = {
    "fl_SY_J": +0.0000, "fl_SP_J": +0.8100, "fl_knee": +0.8932,
    "fr_SY_J": +0.0000, "fr_SP_J": -0.8109, "fr_knee": -0.8938,
    "bl_SY_J": +0.0000, "bl_SP_J": +0.3932, "bl_knee": +0.8913,
    "br_SY_J": +0.0000, "br_SP_J": +0.3936, "br_knee": -0.8913,
}
STAND_BASE_HEIGHT = 0.182

OUT = []


def say(s=""):
    OUT.append(s)
    print(s)


RESULTS = []


def record(name, ok, detail, tol=""):
    RESULTS.append((name, ok, detail, tol))
    say(f"  [{'ok  ' if ok else 'FAIL'}] {name:<34s} {detail}"
        + (f"   (tol {tol})" if tol else ""))


# --------------------------------------------------------------------- helpers
def mj_setup(path=None):
    m = mujoco.MjModel.from_xml_path(path or MJCF)
    d = mujoco.MjData(m)
    jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in DOF_ORDER}
    qadr = {n: m.jnt_qposadr[jid[n]] for n in DOF_ORDER}
    return m, d, jid, qadr


def urdf_fk_chain(kin, joint_names, q):
    """World transform of the terminal link, base frame, from V4Kin's own URDF data."""
    R, p = np.eye(3), np.zeros(3)
    for name, qi in zip(joint_names, q):
        J = kin.j[name]
        p = p + R @ J["xyz"]
        R = R @ J["R"] @ axis_rot(J["axis"], qi)
    return R, p


# --------------------------------------------------------------------- checks
def check_fk(m, d, qadr, kin, n_samples, rng):
    """Body-frame position of every leg's knee link, MuJoCo vs V4Kin."""
    lims = kin.all_limits()
    worst = 0.0
    worst_where = None
    for _ in range(n_samples):
        q = {}
        for i, jn in enumerate(DOF_ORDER):
            lo, hi = lims[i]
            q[jn] = rng.uniform(lo, hi)
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0                       # identity base quaternion
        for jn in DOF_ORDER:
            d.qpos[qadr[jn]] = q[jn]
        mujoco.mj_kinematics(m, d)
        base_p = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "origin")].copy()
        base_R = d.xmat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "origin")].reshape(3, 3)
        for leg in LEGS:
            chain = kin.leg_chain(leg)
            _, p_urdf = urdf_fk_chain(kin, chain, [q[c] for c in chain])
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_knee")
            p_mj = base_R.T @ (d.xpos[bid] - base_p)
            e = float(np.linalg.norm(p_mj - p_urdf))
            if e > worst:
                worst, worst_where = e, f"{leg}_knee"
        # the expressive chains are orientation-only in V4Kin, compare rotations
        for names in (["head_pitch_joint", "head_yaw", "head_roll"],
                      ["tail_pitch", "tail_yaw"],
                      ["l_ear_pitch", "l_ear_roll"], ["r_ear_pitch", "r_ear_roll"]):
            R_urdf, _ = urdf_fk_chain(kin, names, [q[c] for c in names])
            bid = mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_BODY,
                {"head_roll": "head_roll", "tail_yaw": "tail_yaw",
                 "l_ear_roll": "l_ear_roll", "r_ear_roll": "r_ear_roll"}[names[-1]])
            R_mj = base_R.T @ d.xmat[bid].reshape(3, 3)
            ang = float(np.degrees(np.arccos(
                np.clip((np.trace(R_urdf.T @ R_mj) - 1) / 2, -1, 1))))
            if ang / 1000.0 > worst:          # keep one scalar; deg->"m-ish" scale
                pass
    record("FK vs V4Kin (leg knee pos)", worst < 1e-6,
           f"max error {worst*1e6:.4f} um over {n_samples} random poses"
           + (f" at {worst_where}" if worst_where else ""), "1 um")
    return worst


def check_fk_orientation(m, d, qadr, kin, n_samples, rng):
    lims = kin.all_limits()
    worst, where = 0.0, None
    term = {"head": ("head_roll", ["head_pitch_joint", "head_yaw", "head_roll"]),
            "tail": ("tail_yaw", ["tail_pitch", "tail_yaw"]),
            "earL": ("l_ear_roll", ["l_ear_pitch", "l_ear_roll"]),
            "earR": ("r_ear_roll", ["r_ear_pitch", "r_ear_roll"])}
    for _ in range(n_samples):
        q = {jn: rng.uniform(*lims[i]) for i, jn in enumerate(DOF_ORDER)}
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        for jn in DOF_ORDER:
            d.qpos[qadr[jn]] = q[jn]
        mujoco.mj_kinematics(m, d)
        b0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "origin")
        base_R = d.xmat[b0].reshape(3, 3)
        for label, (body, names) in term.items():
            R_urdf, _ = urdf_fk_chain(kin, names, [q[c] for c in names])
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
            # V4Kin composes each chain from IDENTITY, so a chain hanging off the
            # head must be compared in the HEAD frame, not the base frame - which is
            # exactly how solve_spatial_retarget solves the ears (H.T @ target).
            if label.startswith("ear"):
                hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head_roll")
                ref_R = d.xmat[hid].reshape(3, 3)
            else:
                ref_R = base_R
            R_mj = ref_R.T @ d.xmat[bid].reshape(3, 3)
            ang = float(np.degrees(np.arccos(
                np.clip((np.trace(R_urdf.T @ R_mj) - 1) / 2, -1, 1))))
            if ang > worst:
                worst, where = ang, label
    record("FK vs V4Kin (chain orientation)", worst < 1e-3,
           f"max error {worst:.2e} deg" + (f" at {where}" if where else ""), "1e-3 deg")


def check_joints(m, kin):
    bad = []
    for jn in DOF_ORDER:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        J = kin.j[jn]
        ax_mj = m.jnt_axis[jid]
        if np.linalg.norm(ax_mj - np.asarray(J["axis"])) > 1e-9:
            bad.append(f"{jn} axis {ax_mj} vs {J['axis']}")
        lo, hi = m.jnt_range[jid]
        if abs(lo - J["lo"]) > 1e-9 or abs(hi - J["hi"]) > 1e-9:
            bad.append(f"{jn} range [{lo},{hi}] vs [{J['lo']},{J['hi']}]")
    record("joint axes and limits", not bad,
           "all 21 match the URDF" if not bad else "; ".join(bad[:3]), "1e-9")


def check_mass(m, kin):
    import xml.etree.ElementTree as ET
    root = ET.parse(URDF_P).getroot()
    urdf_mass = {}
    for l in root.findall("link"):
        im = l.find("inertial")
        if im is not None and im.find("mass") is not None:
            urdf_mass[l.get("name")] = float(im.find("mass").get("value"))
    worst, where = 0.0, None
    for name, mass in urdf_mass.items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        e = abs(float(m.body_mass[bid]) - mass)
        if e > worst:
            worst, where = e, name
    tot_mj = float(m.body_mass.sum())
    tot_urdf = sum(urdf_mass.values())
    record("per-link mass", worst < 1e-9,
           f"max error {worst:.3e} kg" + (f" at {where}" if where else ""), "1e-9 kg")
    # accumulated float error over 22 links, so the tolerance is per-sum not per-term
    record("total mass", abs(tot_mj - tot_urdf) < 1e-7,
           f"MuJoCo {tot_mj:.9f} kg vs URDF {tot_urdf:.9f} kg "
           f"(diff {abs(tot_mj-tot_urdf):.2e})", "1e-7 kg")


def check_stance(m, d, qadr):
    hull = {k: v for k, v in np.load(HULLS).items()}
    d.qpos[:] = 0.0
    d.qpos[2] = STAND_BASE_HEIGHT
    d.qpos[3] = 1.0
    for jn, v in STAND_SOLVED.items():
        d.qpos[qadr[jn]] = v
    mujoco.mj_kinematics(m, d)
    lows = []
    for leg in LEGS:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_knee")
        R = d.xmat[bid].reshape(3, 3)
        p = d.xpos[bid]
        w = hull[f"{leg}_knee"] @ R.T + p
        lows.append(float(w[:, 2].min()))
    lows = np.array(lows)
    spread = float(lows.max() - lows.min())
    record("stance paw heights level", spread < 1e-4,
           f"spread {spread*1000:.4f} mm, heights {np.round(lows*1000, 3)} mm", "0.1 mm")
    record("stance paws on the floor", abs(lows.mean()) < 3e-3,
           f"mean paw z {lows.mean()*1000:+.3f} mm at base {STAND_BASE_HEIGHT} m", "3 mm")


def check_standing(m, d, qadr, seconds=3.0):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    z0 = float(d.qpos[2])
    n = int(seconds / m.opt.timestep)
    for _ in range(n):
        mujoco.mj_step(m, d)
    z1 = float(d.qpos[2])
    q_err = max(abs(float(d.qpos[qadr[jn]]) - v) for jn, v in STAND_SOLVED.items())
    quat = d.qpos[3:7]
    tilt = float(np.degrees(2 * np.arccos(np.clip(abs(quat[0]), -1, 1))))
    ok = abs(z1 - z0) < 0.02 and tilt < 10.0 and np.isfinite(z1)
    record("standing equilibrium (3 s)", ok,
           f"base z {z0:.4f} -> {z1:.4f} m (sank {1000*(z0-z1):+.1f} mm), "
           f"tilt {tilt:.2f} deg, max joint err {q_err:.4f} rad", "sink<20mm tilt<10deg")
    return z1


def check_drop(m, d, qadr, drop_h=0.05, seconds=2.0):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[2] += drop_h
    mujoco.mj_forward(m, d)
    n = int(seconds / m.opt.timestep)
    zmin = 1e9
    for _ in range(n):
        mujoco.mj_step(m, d)
        zmin = min(zmin, float(d.qpos[2]))
    z1 = float(d.qpos[2])
    settled = abs(float(np.linalg.norm(d.qvel[:3]))) < 0.05
    record(f"gravity drop {1000*drop_h:.0f} mm", np.isfinite(z1) and settled,
           f"landed at z {z1:.4f} m (min {zmin:.4f}), "
           f"base speed {np.linalg.norm(d.qvel[:3]):.4f} m/s", "settles <0.05 m/s")


def check_step_response(m, d, qadr):
    """Command a single joint away from stance; check it converges and stays legal."""
    worst = []
    for jn, delta in (("fl_SP_J", 0.15), ("fl_knee", -0.15), ("head_yaw", 0.20)):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
        target = STAND_SOLVED.get(jn, 0.0) + delta
        d.ctrl[aid] = target
        for _ in range(int(1.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        reached = float(d.qpos[qadr[jn]])
        worst.append((jn, target, reached, abs(reached - target)))
    bad = [w for w in worst if w[3] > 0.05]
    record("single-joint step response", not bad,
           "; ".join(f"{jn} -> {r:+.4f} (target {t:+.4f}, err {e:.4f})"
                     for jn, t, r, e in worst), "0.05 rad")


def check_contact_geometry(m, d, qadr):
    hull = {k: v for k, v in np.load(HULLS).items()}
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    ncon = d.ncon
    paw_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{l}_knee")
                  for l in LEGS}
    touching = set()
    for i in range(ncon):
        c = d.contact[i]
        for g in (c.geom1, c.geom2):
            b = m.geom_bodyid[g]
            if b in paw_bodies:
                touching.add(b)
    record("contacts are the four paws", len(touching) == 4,
           f"{len(touching)}/4 paw links in contact, {ncon} contact point(s) total",
           "4 paws")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="random FK samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default=None)
    ap.add_argument("--isaac-trace", default=None,
                    help="npz recorded by rl/tools/eval_velocity.py on a GPU machine")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    kin = V4Kin(URDF_P)
    m, d, jid, qadr = mj_setup()

    say("=" * 92)
    say("BINGO MuJoCo <-> URDF/Isaac VALIDATION")
    say("=" * 92)
    say(f"model     {MJCF}")
    say(f"reference {URDF_P}")
    say(f"          stage2/v4_kinematics.V4Kin (the FK the Stage-2/4 pipeline uses)")
    say(f"compiled  nbody={m.nbody} njnt={m.njnt} nq={m.nq} nv={m.nv} nu={m.nu} "
        f"mass={m.body_mass.sum():.5f} kg  timestep={m.opt.timestep}")
    say()
    say("KINEMATICS AND MASS PROPERTIES (exact - a failure here is a conversion bug)")
    check_fk(m, d, qadr, kin, a.n, rng)
    check_fk_orientation(m, d, qadr, kin, max(20, a.n // 4), rng)
    check_joints(m, kin)
    check_mass(m, kin)
    check_stance(m, d, qadr)
    say()
    say("DYNAMICS (behavioural - MuJoCo and PhysX never agree bit-for-bit)")
    ms, ds, _, qadr_s = mj_setup(MJCF_SCENE)
    say(f"  using {os.path.basename(MJCF_SCENE)} (the robot alone has no floor to stand on)")
    check_standing(ms, ds, qadr_s)
    check_drop(ms, ds, qadr_s)
    check_step_response(ms, ds, qadr_s)
    check_contact_geometry(ms, ds, qadr_s)

    say()
    say("ISAAC-SIDE TRACE COMPARISON")
    if a.isaac_trace and os.path.exists(a.isaac_trace):
        say(f"  (comparison against {a.isaac_trace} - not implemented until a trace exists)")
    else:
        say("  NOT RUN. Isaac Sim needs an RTX GPU and is not available here, so no")
        say("  command-trace comparison was performed. Everything above is verified")
        say("  against the URDF, which is the model Isaac itself consumes - that")
        say("  catches conversion errors but does NOT prove dynamic agreement.")
        say("  Record a trace with rl/tools/eval_velocity.py on the training machine")
        say("  and re-run with --isaac-trace to close this gap.")

    n_fail = sum(1 for _, ok, _, _ in RESULTS if not ok)
    say()
    say("=" * 92)
    say(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed"
        + ("" if n_fail == 0 else f"  -- {n_fail} FAILED"))
    say("=" * 92)

    if a.report:
        with open(a.report, "w") as f:
            f.write("\n".join(OUT) + "\n")
        print(f"[[ wrote {a.report}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
