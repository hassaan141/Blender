"""Phase 2 retargeter: dog trot (Peng/Laikago clip) -> Bingo AMP motion .npz.

Joint-space affine retarget. The dog clip is already in a per-leg quadruped joint
convention (abduction, thigh, knee). We:
  1. Parse dog_trot.txt (Frames = [root_pos(3), root_quat xyzw(4), 12 joints]).
  2. Map each dog leg -> Bingo leg (FR->fr, FL->fl, RR->br, RL->bl).
  3. Affine-map (abduction,thigh,knee) -> Bingo canonical (SY,SP,knee) centered on
     Bingo's crouch, amplitude scaled to fit Bingo joint limits. This preserves the
     dog's PHASING / diagonal coordination (the actual "style"), which is what the
     AMP discriminator keys on.
  4. Apply Bingo's verified per-leg AXIS SIGN MAP so the flipped-axis right legs
     (fr: SP&knee, br: knee) mirror correctly -> symmetric, no inverted leg.
  5. Fill the IsaacLab motion schema: dof_pos/vel + body_pos/rot/lin+ang vel for the
     reference body ("origin") and the 4 feet ("*_knee"), via analytical FK. numpy only.

Output: bingo_trot.npz next to this script's --out path.
Run: python retarget_dog_to_bingo.py --dog <dog_trot.txt> --out <bingo_trot.npz> [--cycles N]
"""

import argparse
import json
import xml.etree.ElementTree as ET

import numpy as np

# ----- Bingo per-leg axis SIGN MAP (verified via FK against the URDF) -----
# canonical value -> URDF joint value. Left legs are reference (+1).
# SP axes: fl+y bl+y fr-y br+y ; knee axes: fl-y bl-y fr+y br+y
SP_SIGN = {"fl": +1.0, "bl": +1.0, "fr": -1.0, "br": +1.0}
KN_SIGN = {"fl": +1.0, "bl": +1.0, "fr": -1.0, "br": -1.0}
# SY axes: fl-x bl+x fr+x br-x -> pick signs so a +canonical abducts outward
SY_SIGN = {"fl": -1.0, "bl": +1.0, "fr": +1.0, "br": -1.0}

# canonical crouch centers (match v7 default) and amplitude gains
SP0, KN0 = -0.30, 0.60
# bigger SP/knee swing -> more leg reach & lift (taller, more dynamic gait); tiny SY
# so legs stay UNDER the body instead of splaying out to the sides.
# v11: split SP/knee swing PER LEG-PAIR. v10 used a global G_SP/G_KN, which made the
# front legs over-reach (~2x the back's foot-z lift) and see-saw the body into a choppy
# forward-lunge. The dog clip's front thigh swing is naturally larger, so we cut the
# FRONT (fl,fr) gains ~40% to match the back's ~50mm foot-z p2p; back (bl,br) unchanged.
G_SY = 0.15
G_SP_LEG = {"fl": 0.36, "fr": 0.36, "bl": 0.60, "br": 0.60}
G_KN_LEG = {"fl": 0.51, "fr": 0.51, "bl": 0.85, "br": 0.85}
# phase sign of thigh->SP and knee-fold->knee (flip if replay looks backward)
SP_PHASE = +1.0
KN_PHASE = +1.0

# --- Naturalness fixes (bingoAMPNaturalMovement.md) ---
# Fix 1: the AMP "foot" was the knee HINGE (knee angle hardcoded to 0 in FK), which
# barely moves vertically. The real paw lift lives on the SHANK below the hinge. We now
# FK with the real per-frame knee angle and add a downward offset along the shank to the
# contact point, so the reference feet show true vertical travel. The SAME offset is
# applied on the policy side in bingo_amp_env.py (hinge + rotated offset) so the AMP
# foot-height channel is matched on both sides (else the discriminator classifies for
# free on that channel and the style gradient collapses).
SHANK_LEN = 0.12  # m, offset from knee hinge down the shank to the contact point
# Fix 2: carry the dog clip's real root vertical bob + pitch/roll (the retargeter used to
# throw these away for a flat glide + frozen identity quat). Angle gain damps the dog's
# tilt to Bingo's stiffer body; clamp keeps the reference sane.
# v14: zero the injected pitch/roll bob (it caused the see-saw/choppiness the user
# disliked; v9's flat orientation was preferred). Naturalness now comes from the multi-gait
# reference distribution + AMP, not from a synthesized body bob. The subtle vertical z-bob
# (from the dog root height) is kept separately below; only pitch/roll is zeroed here.
G_ROOT_ROT = 0.0
ROOT_ROT_CLAMP = 0.06  # rad

BINGO_HEIGHT = 0.19  # target base height (v7 walking height)
DOG2BINGO = None     # horizontal scale, computed from leg-length ratio below

DOG_LEG = {"FR": "fr", "FL": "fl", "RR": "br", "RL": "bl"}
DOG_ORDER = ["FR", "FL", "RR", "RL"]
BINGO_LEGS = ["fl", "fr", "bl", "br"]
JOINT_ORDER = [f"{l}_{j}" for l in BINGO_LEGS for j in ("SY_J", "SP_J", "knee")]  # 12
BODY_NAMES = ["origin", "fl_knee", "fr_knee", "bl_knee", "br_knee"]


def rrpy(rpy):
    r, p, y = rpy
    cx, sx = np.cos(r), np.sin(r); cy, sy = np.cos(p), np.sin(p); cz, sz = np.cos(y), np.sin(y)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def raxis(a, ang):
    a = a / (np.linalg.norm(a) + 1e-12); x, y, z = a; c, s = np.cos(ang), np.sin(ang); C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def load_urdf(path):
    root = ET.parse(path).getroot()
    J = {}
    for j in root.findall("joint"):
        o = j.find("origin"); ax = j.find("axis")
        J[j.get("name")] = dict(
            xyz=np.array([float(v) for v in (o.get("xyz").split() if o is not None else [0, 0, 0])]),
            rpy=np.array([float(v) for v in (o.get("rpy").split() if (o is not None and o.get("rpy")) else [0, 0, 0])]),
            axis=np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3),
        )
    return J


def foot_tip_base(J, leg, sy, sp, kn):
    """Shank-TIP (foot contact point) in base frame for given SY,SP,knee.
    Fix 1: include the real knee angle and step SHANK_LEN down the shank so the foot
    lifts as the knee flexes (the hinge alone barely moves vertically)."""
    p = np.zeros(3); R = np.eye(3)
    for jt, ang in [(f"{leg}_SY_J", sy), (f"{leg}_SP_J", sp), (f"{leg}_knee", kn)]:
        j = J[jt]; p = p + R @ j["xyz"]; R = R @ rrpy(j["rpy"]); R = R @ raxis(j["axis"], ang)
    return p + R @ np.array([0.0, 0.0, -SHANK_LEN])


def quat_to_euler_xyzw(q):
    """[T,4] xyzw -> roll,pitch,yaw arrays."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def euler_to_quat_wxyz(roll, pitch, yaw):
    """roll,pitch,yaw arrays -> [T,4] wxyz."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], axis=-1)


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dog", required=True)
    ap.add_argument("--urdf", default="/home/hassaan/Bingo/Blender/URDF/bingo_urdf_rev_3/urdf/bingo_urdf_rev_3_real_values.urdf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cycles", type=int, default=6, help="repeat the gait cycle N times for a longer clip")
    ap.add_argument("--fwd_speed", type=float, default=None, help="override root forward speed (m/s)")
    ap.add_argument("--no_symmetrize", action="store_true",
                    help="disable v13 L/R symmetrization (reproduce the asymmetric v12 reference)")
    args = ap.parse_args()

    d = json.load(open(args.dog))
    F = np.array(d["Frames"])
    fps = 1.0 / d["FrameDuration"]
    T = len(F)
    joints = F[:, 7:19].reshape(T, 4, 3)  # [T,4,{abd,thigh,knee}] order FR,FL,RR,RL
    root_x = F[:, 0]  # dog forward
    J = load_urdf(args.urdf)

    # global centering stats (across all legs) to preserve relative variation
    abd_m = joints[:, :, 0].mean(); th_m = joints[:, :, 1].mean(); kn_m = joints[:, :, 2].mean()

    # forward speed from dog clip, scaled to Bingo (leg-length ratio ~ 0.16/0.35)
    scale = 0.16 / 0.35
    fwd_disp = (root_x[-1] - root_x[0]) * scale
    cyc_time = T / fps
    fwd_speed = fwd_disp / cyc_time
    if args.fwd_speed is not None:
        fwd_speed = args.fwd_speed  # align reference root speed with the task target

    # ---- build per-leg CANONICAL joint trajectories (sign map applied later) ----
    cano = {}  # leg -> [cSY, cSP, cKN], each length T, in the common canonical frame
    for di, dl in enumerate(DOG_ORDER):
        leg = DOG_LEG[dl]
        abd, th, kn = joints[:, di, 0], joints[:, di, 1], joints[:, di, 2]
        cSY = G_SY * (abd - abd_m)
        cSP = SP0 + SP_PHASE * G_SP_LEG[leg] * (th - th_m)
        cKN = KN0 + KN_PHASE * G_KN_LEG[leg] * (kn_m - kn)  # more dog-fold (neg) -> more Bingo-fold (pos)
        cano[leg] = [cSY, cSP, cKN]

    # ---- v13: enforce LEFT/RIGHT symmetry ----
    # The raw dog clip isn't perfectly L/R-symmetric over one cycle, which left the
    # front-left leg under-swinging (loose fl.br diagonal, +0.81, while fr.bl was +0.99).
    # A trot has the diagonal pairs IN phase (fl~br, fr~bl) with the two pairs a half-cycle
    # apart (fl.fr, fl.bl anti-phase) -- confirmed by the v12 eval. So rebuild each leg from
    # a single per-END (front/back) canonical waveform + a half-cycle phase offset: phase-0
    # legs = fl, br; phase-0.5 legs = fr, bl. Now left and right swing identically. Done in
    # canonical space (before the sign map) so it's axis-agnostic.
    if not args.no_symmetrize:
        h = T // 2  # half-cycle shift, in frames (clip is one gait cycle of length T)
        for j in range(3):  # SY, SP, KN
            front = 0.5 * (cano["fl"][j] + np.roll(cano["fr"][j], h))  # phase-0 front waveform
            back = 0.5 * (cano["br"][j] + np.roll(cano["bl"][j], h))   # phase-0 back waveform
            cano["fl"][j], cano["fr"][j] = front, np.roll(front, -h)
            cano["br"][j], cano["bl"][j] = back, np.roll(back, -h)

    # ---- apply the per-leg axis SIGN MAP -> URDF dof ----
    dof = np.zeros((T, 12))
    jidx = {n: i for i, n in enumerate(JOINT_ORDER)}
    for leg in BINGO_LEGS:
        cSY, cSP, cKN = cano[leg]
        dof[:, jidx[f"{leg}_SY_J"]] = np.clip(SY_SIGN[leg] * cSY, -0.42, 0.42)
        dof[:, jidx[f"{leg}_SP_J"]] = np.clip(SP_SIGN[leg] * cSP, -1.55, 1.55)
        dof[:, jidx[f"{leg}_knee"]] = np.clip(KN_SIGN[leg] * cKN, -1.55, 1.55)

    # ---- Fix 2: real root trajectory (bob + pitch/roll) from the dog clip ----
    # advance forward at fwd_speed; height oscillates around BINGO_HEIGHT with the dog's
    # (scaled) vertical bob; orientation carries the dog's pitch/roll (yaw dropped -> we
    # travel straight +x). The old retargeter used a flat height + frozen identity quat,
    # killing 3 of the AMP "naturalness" channels (root_h, root_quat, root_ang_vel).
    t = np.arange(T) / fps
    dt = 1.0 / fps
    z_bob = (F[:, 2] - F[:, 2].mean()) * scale
    roll, pitch, _ = quat_to_euler_xyzw(F[:, 3:7])
    roll = np.clip(G_ROOT_ROT * (roll - roll.mean()), -ROOT_ROT_CLAMP, ROOT_ROT_CLAMP)
    pitch = np.clip(G_ROOT_ROT * (pitch - pitch.mean()), -ROOT_ROT_CLAMP, ROOT_ROT_CLAMP)

    root_pos = np.zeros((T, 3)); root_pos[:, 0] = fwd_speed * t; root_pos[:, 2] = BINGO_HEIGHT + z_bob
    root_quat = euler_to_quat_wxyz(roll, pitch, np.zeros(T))          # [T,4] wxyz
    R_root = np.stack([quat_wxyz_to_R(q) for q in root_quat])         # [T,3,3]
    root_ang = np.zeros((T, 3))
    root_ang[:, 0] = np.gradient(roll, dt); root_ang[:, 1] = np.gradient(pitch, dt)

    # ---- FK feet (shank TIP, Fix 1) in base frame, placed in world via the root pose ----
    nb = len(BODY_NAMES)
    body_pos = np.zeros((T, nb, 3)); body_rot = np.zeros((T, nb, 4))
    body_pos[:, 0] = root_pos; body_rot[:, 0] = root_quat
    for bi, bn in enumerate(BODY_NAMES[1:], start=1):
        leg = bn.split("_")[0]
        body_rot[:, bi] = root_quat  # feet orientation is unused by the obs; keep consistent
        for ti in range(T):
            fb = foot_tip_base(J, leg, dof[ti, jidx[f"{leg}_SY_J"]],
                               dof[ti, jidx[f"{leg}_SP_J"]], dof[ti, jidx[f"{leg}_knee"]])
            body_pos[ti, bi] = root_pos[ti] + R_root[ti] @ fb

    # ---- repeat cycles for a longer, loopable clip ----
    if args.cycles > 1:
        dof = np.tile(dof, (args.cycles, 1))
        reps = []
        for c in range(args.cycles):
            rp = root_pos.copy(); rp[:, 0] += c * fwd_speed * cyc_time
            reps.append(rp)
        root_pos = np.concatenate(reps, 0)
        root_quat = np.tile(root_quat, (args.cycles, 1))
        root_ang = np.tile(root_ang, (args.cycles, 1))
        bp = []
        for c in range(args.cycles):
            b = body_pos.copy(); b[:, :, 0] += c * fwd_speed * cyc_time
            bp.append(b)
        body_pos = np.concatenate(bp, 0)
        body_rot = np.tile(body_rot, (args.cycles, 1, 1))
        t = np.arange(len(dof)) / fps

    # ---- velocities by finite difference ----
    dof_vel = np.gradient(dof, dt, axis=0)
    body_lin = np.gradient(body_pos, dt, axis=0)
    body_ang = np.zeros((len(dof), nb, 3))  # feet ang-vel unused; root carries the real bob
    body_ang[:, 0] = root_ang  # already tiled to len(dof) in the cycles block above

    np.savez(
        args.out,
        fps=np.array(fps),
        dof_names=np.array(JOINT_ORDER),
        body_names=np.array(BODY_NAMES),
        dof_positions=dof.astype(np.float32),
        dof_velocities=dof_vel.astype(np.float32),
        body_positions=body_pos.astype(np.float32),
        body_rotations=body_rot.astype(np.float32),
        body_linear_velocities=body_lin.astype(np.float32),
        body_angular_velocities=body_ang.astype(np.float32),
    )
    print(f"saved {args.out}: {len(dof)} frames @ {fps:.1f}fps, fwd_speed={fwd_speed:.3f} m/s")
    print("dof range per joint:")
    for i, n in enumerate(JOINT_ORDER):
        print(f"  {n:10s} [{dof[:,i].min():+.2f}, {dof[:,i].max():+.2f}]")
    print("foot z rel root (Fix 1: p2p should now be ~0.03-0.06 m, was ~0.008):")
    for bi, bn in enumerate(BODY_NAMES[1:], start=1):
        fz = (body_pos[:, bi, 2] - body_pos[:, 0, 2])
        print(f"  {bn:8s} z rel root [{fz.min():+.3f}, {fz.max():+.3f}]  p2p {fz.max()-fz.min():.3f}")
    print("root naturalness channels (Fix 2, were dead constants):")
    print(f"  root height   [{root_pos[:,2].min():.3f}, {root_pos[:,2].max():.3f}]  p2p {root_pos[:,2].ptp():.3f}")
    print(f"  root roll     p2p {roll.ptp():.3f} rad   pitch p2p {pitch.ptp():.3f} rad")
    print(f"  root ang-vel  |max| {np.abs(body_ang[:,0]).max():.3f} rad/s")


if __name__ == "__main__":
    main()
