#!/usr/bin/env python3
"""Build a skill-tracking reference: stand -> clip -> stand, at the 24 Hz control rate.

    frames [0, TIN)                blend stand -> clip frame 0   (entering)
    frames [TIN, TIN+T)            the authored Stage-4 clip
    frames [TIN+T, TIN+T+TOUT)     blend last clip frame -> stand (exiting)
    frames [.., +HOLD)             stand

The clip is re-expressed so frame 0 sits at the origin facing +x (the runtime
re-anchors it to the robot's pose when the skill starts). Paw tips are computed by
MuJoCo forward kinematics on the browser MJCF, so reference and simulation use the
same geometry. Usage: skill_ref.py timid  ->  tools/skill_track/out/timid_ref.{npz,json}
"""
import json, sys
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
MJCF = ROOT / "bingo-simulator/app/public/robot/bingo_scene.xml"
OUT = Path(__file__).resolve().parent / "out"
JOINTS = ["fl_SY_J", "fl_SP_J", "fl_knee", "fr_SY_J", "fr_SP_J", "fr_knee",
          "bl_SY_J", "bl_SP_J", "bl_knee", "br_SY_J", "br_SP_J", "br_knee",
          "head_pitch_joint", "head_yaw", "head_roll", "tail_pitch", "tail_yaw",
          "l_ear_pitch", "l_ear_roll", "r_ear_pitch", "r_ear_roll"]
PAWS = ["fl_knee", "fr_knee", "bl_knee", "br_knee"]
TIP = np.array([0., 0., -.12])        # paw tip in the knee body frame (browser buildCommandObs)
TIN, TOUT, HOLD = 24, 24, 12          # 1 s in, 1 s out, 0.5 s stand


def yaw_of(q): return np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
def qz(yaw): return np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])
def qrot(q, v):
    t = 2*np.cross(q[1:], v); return v + q[0]*t + np.cross(q[1:], t)
def slerp(a, b, w):
    d = np.dot(a, b)
    if d < 0: b, d = -b, -d
    if d > .9995: q = a + w*(b-a); return q/np.linalg.norm(q)
    th = np.arccos(d); return (np.sin((1-w)*th)*a + np.sin(w*th)*b)/np.sin(th)


def build(name):
    z = np.load(ROOT / f"motions/{name}_v4.npz")
    names = list(z["dof_names"]); idx = [names.index(n) for n in JOINTS]
    q = z["dof_positions"][:, idx].astype(np.float64)
    p = z["body_positions"][:, 0].astype(np.float64).copy(); r = z["body_rotations"][:, 0].astype(np.float64)
    contacts = z["contacts"].astype(np.float64)
    # Anchor: frame 0 at xy = 0, heading 0.
    yaw0 = yaw_of(r[0]); qi = qz(-yaw0); p[:, :2] -= p[0, :2]
    p = np.array([qrot(qi, x) for x in p]); r = np.array([qmul(qi, x) for x in r])

    m = mujoco.MjModel.from_xml_path(str(MJCF)); d = mujoco.MjData(m)
    stand = m.key_qpos[0].copy()
    qa = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINTS]
    s_q = stand[qa]; s_z = stand[2]
    pid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in PAWS]
    def tips_at(pos, quat, jq):
        d.qpos[:] = stand; d.qpos[:3] = pos; d.qpos[3:7] = quat
        for j, a in enumerate(qa): d.qpos[a] = jq[j]
        mujoco.mj_kinematics(m, d)
        return np.array([d.xpos[b] + qrot(d.xquat[b], TIP) for b in pid])
    # The Isaac reference sits ~2 cm lower than the MJCF's paw geometry. Shift the clip's
    # root height by one constant so in-contact paws are at the MJCF standing tip height.
    stand_tip = tips_at(stand[:3], stand[3:7], s_q)[:, 2].min()
    low = np.array([tips_at(p[i], r[i], q[i])[:, 2][contacts[i] > .5].min() for i in range(len(q)) if contacts[i].any()])
    z_offset = stand_tip - np.median(low); p[:, 2] += z_offset
    # Keep the paws planted through the transitions: place the clip so its frame-0 paw
    # centroid is under the standing paw centroid, and stand at the end where the last
    # clip frame's paws are (heading = final clip heading).
    cen = lambda pos, quat, jq: tips_at(pos, quat, jq)[:, :2].mean(0)
    stand_cen = cen(stand[:3], stand[3:7], s_q)                       # relative to root at origin
    p[:, :2] += stand_cen - cen(p[0], r[0], q[0])
    yT = yaw_of(r[-1]); end_cen = cen(p[-1], r[-1], q[-1])
    end_xy = end_cen - qrot(qz(yT), np.array([*stand_cen, 0.]))[:2]
    T = len(q); N = TIN + T + TOUT + HOLD
    legs = np.zeros((N, 12)); expr = np.zeros((N, 9)); rp = np.zeros((N, 3)); rq = np.zeros((N, 4)); ct = np.ones((N, 4))
    for k in range(N):
        if k < TIN:
            w = k / TIN; qq = (1-w)*s_q + w*q[0]
            rp[k] = [*((1-w)*np.zeros(2) + w*p[0, :2]), (1-w)*s_z + w*p[0, 2]]; rq[k] = slerp(qz(yaw_of(r[0])), r[0], w)
        elif k < TIN + T:
            i = k - TIN; qq = q[i]; rp[k] = p[i]; rq[k] = r[i]; ct[k] = contacts[i]
        elif k < TIN + T + TOUT:
            w = (k - TIN - T + 1) / TOUT; qq = (1-w)*q[-1] + w*s_q
            rp[k] = [*((1-w)*p[-1, :2] + w*end_xy), (1-w)*p[-1, 2] + w*s_z]; rq[k] = slerp(r[-1], qz(yaw_of(r[-1])), w)
        else:
            qq = s_q; rp[k] = [*end_xy, s_z]; rq[k] = qz(yaw_of(r[-1]))
        legs[k] = qq[:12]; expr[k] = qq[12:]
    # Entering expression blend starts from the robot's live expression; the runtime
    # overrides frames [0, TIN) with lerp(start, expr[TIN], k/TIN). Stored here from 0.
    tips = np.array([tips_at(rp[k], rq[k], np.concatenate([legs[k], expr[k]])) for k in range(N)])
    lowest = tips[:, :, 2].min(1)
    meta = {"name": name, "fps": 24.0, "frames": N, "tin": TIN, "clip_frames": T, "tout": TOUT, "hold": HOLD,
            "joint_order": JOINTS, "source": f"motions/{name}_v4.npz", "root_z_offset": float(z_offset), "stand_tip_z": float(stand_tip), "mjcf": str(MJCF.relative_to(ROOT))}
    return meta, dict(legs=legs, expr=expr, root_pos=rp, root_quat=rq, tips=tips, contacts=ct), lowest


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "timid"
    meta, ref, lowest = build(name)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}_ref.npz", meta=json.dumps(meta), **ref)
    (OUT / f"{name}_ref.json").write_text(json.dumps({**meta, **{k: np.round(v, 6).tolist() for k, v in ref.items()}}))
    t = meta["tin"]; c = meta["clip_frames"]
    print(f"{name}: {meta['frames']} frames; lowest paw-tip z clip min/median {lowest[t:t+c].min():.4f}/{np.median(lowest[t:t+c]):.4f}"
          f"; stand frames {lowest[0]:.4f}/{lowest[-1]:.4f}")
