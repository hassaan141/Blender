"""Stage 2.11 - quantitative evaluation of the Cheeky retarget + report.

Compares the solved v4 motion against Ashley's extracted source motion in a
common, scale-normalised frame, and writes CHEEKY_RETARGET_REPORT.md.

Metrics: per-foot trajectory error, planted-foot sliding, contact preservation,
root deviation, knee error, ground penetration, joint-limit saturation, and the
largest-error frames.

Run (system python):
  python3 stage2/evaluate_retarget.py \
      --keypoints stage2/out/cheeky_source_keypoints.npz \
      --contacts  stage2/out/cheeky_contacts.npz \
      --retarget  stage2/out/cheeky_v4_retarget.npz \
      --urdf "URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints.urdf" \
      --report CHEEKY_RETARGET_REPORT.md
"""
import argparse, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_kinematics import V4Kin, LEGS, DOF_ORDER, PAW_DROP, quat_to_mat, mat_to_quat, rotvec_of

ALEGS = ["aFL", "aFR", "aBL", "aBR"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoints", required=True)
    ap.add_argument("--contacts", required=True)
    ap.add_argument("--retarget", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    k = np.load(a.keypoints, allow_pickle=True)
    c = np.load(a.contacts, allow_pickle=True)
    r = np.load(a.retarget, allow_pickle=True)
    kin = V4Kin(a.urdf)

    fps = float(r["fps"]); frames = r["frames"]; T = len(frames); dt = 1.0 / fps
    dof = r["dof_positions"].astype(float)
    tips = r["tips_world"].astype(float)
    knees = r["knees_world"].astype(float)
    root = r["root_pos"].astype(float)
    ct = r["contacts"]
    foot_err = r["foot_err"].astype(float)
    legmap = [s for s in r["leg_map"]]
    # robot leg -> ashley leg
    RMAP = {}
    for s in legmap:
        al, rl = s.split("->")
        RMAP[rl] = al

    # --- source in metric, robot frame (reflection + per-leg scale, same as solve) --
    F = np.diag([-1.0, 1.0, 1.0])
    rf = lambda P: P @ F
    rest = {l: k["rest_lengths"][i] for i, l in enumerate(ALEGS)}
    s_leg = {rl: kin.leg_reach(rl) / float(rest[RMAP[rl]].sum()) for rl in LEGS}
    paw_src = {al: rf(k[f"paw_{al}"].astype(float)) for al in ALEGS}
    hip_src = {al: rf(k[f"hip_{al}"].astype(float)) for al in ALEGS}
    knee_src = {al: rf(k[f"knee_{al}"].astype(float)) for al in ALEGS}
    body_src = rf(k["body_pos"].astype(float))

    # Same once-only anatomical-axis alignment used by the corrected solver.
    urdf_sy = {rl: kin.sy_pos(rl) for rl in LEGS}
    def cen(robot_legs):
        return np.mean([hip_src[RMAP[rl]][0] for rl in robot_legs], 0)
    fwd0 = cen(["fl", "fr"]) - cen(["bl", "br"]); fwd0 /= np.linalg.norm(fwd0)
    lft0 = cen(["fl", "bl"]) - cen(["fr", "br"]); lft0 /= np.linalg.norm(lft0)
    up0 = np.cross(fwd0, lft0); up0 /= np.linalg.norm(up0)
    lft0 = np.cross(up0, fwd0)
    A = np.column_stack([fwd0, lft0, up0]).T
    Rroot = np.array([quat_to_mat(q) for q in r["root_quat"].astype(float)])

    # Express the SOURCE foot relative to its own hip, scaled - this is the target
    # the retarget was asked to reproduce (morphology-independent). Compare direction
    # & reach of source vs robot foot, both measured from the hip, in metres.
    foot_traj_err = np.zeros((T, 4))
    knee_traj_err = np.zeros((T, 4))
    for i in range(T):
        for kk, rl in enumerate(LEGS):
            al = RMAP[rl]
            src_vec_w = A @ ((paw_src[al][i] - hip_src[al][i]) * s_leg[rl])
            src_vec = Rroot[i].T @ src_vec_w
            # robot foot relative to robot hip (both world): use FK in origin frame
            q = dof[i, 3*kk:3*kk+3]
            tip_o, _, kp_o = kin.leg_fk(rl, q)
            rob_vec = tip_o - kin.sy_pos(rl)
            foot_traj_err[i, kk] = np.linalg.norm(rob_vec - src_vec)
            src_kvec_w = A @ ((knee_src[al][i] - hip_src[al][i]) * s_leg[rl])
            src_kvec = Rroot[i].T @ src_kvec_w
            rob_kvec = kp_o - kin.sy_pos(rl)
            knee_traj_err[i, kk] = np.linalg.norm(rob_kvec - src_kvec)

    # --- planted-foot sliding (world) --------------------------------------------
    slips = []
    for i in range(1, T):
        p = ct[i] & ct[i-1]
        if p.any():
            slips.append(np.linalg.norm(tips[i, p, :2] - tips[i-1, p, :2], axis=1).max())
    slide_mm = 1000 * float(np.mean(slips)) if slips else 0.0
    slide_max = 1000 * float(np.max(slips)) if slips else 0.0

    # --- contact preservation: source vs solved schedule -------------------------
    src_ct = np.zeros((T, 4), bool)
    for kk, rl in enumerate(LEGS):
        src_ct[:, kk] = c["contacts"][:, ALEGS.index(RMAP[rl])]
    preserved = float((src_ct == ct).mean()) * 100

    # --- root deviation: Ashley body vs robot root (both scaled, aligned at t0) ---
    s_master = float(np.mean(list(s_leg.values())))
    src_body = ((body_src - body_src[0]) @ A.T) * s_master
    rob_body = root - root[0]
    root_dev = np.linalg.norm(src_body - rob_body, axis=1)

    Rbody_ref = np.array([F @ quat_to_mat(q) @ F for q in k["body_quat"].astype(float)])
    Rbody_des = np.array([A @ (Rb @ Rbody_ref[0].T) @ A.T for Rb in Rbody_ref])
    root_ori_err = np.array([np.linalg.norm(rotvec_of(Rbody_des[i].T @ Rroot[i])) for i in range(T)])

    # Stance/swing velocity evidence in a common aligned world frame.
    source_paws_world = np.zeros_like(tips)
    for i in range(T):
        for kk, rl in enumerate(LEGS):
            source_paws_world[i, kk] = A @ (paw_src[RMAP[rl]][i] * s_leg[rl])
    src_speed = np.linalg.norm(np.diff(source_paws_world[:, :, :2], axis=0), axis=2) * fps
    tgt_speed = np.linalg.norm(np.diff(tips[:, :, :2], axis=0), axis=2) * fps
    persistent = ct[1:] & ct[:-1]
    swing_pair = ~ct[1:] & ~ct[:-1]
    stance_speed_src = np.array([src_speed[persistent[:,j],j].mean() if persistent[:,j].any() else 0
                                 for j in range(4)])
    stance_speed_tgt = np.array([tgt_speed[persistent[:,j],j].mean() if persistent[:,j].any() else 0
                                 for j in range(4)])
    swing_dist_src = np.array([src_speed[swing_pair[:,j],j].sum()/fps for j in range(4)])
    swing_dist_tgt = np.array([tgt_speed[swing_pair[:,j],j].sum()/fps for j in range(4)])

    # --- ground penetration ------------------------------------------------------
    paw_z = tips[:, :, 2] - PAW_DROP
    pen_mm = -min(0.0, paw_z.min()) * 1000
    pen_frames = int((paw_z < -1e-3).any(1).sum())

    # --- joint-limit saturation --------------------------------------------------
    lims = kin.all_limits()
    sat_rows = []
    for j in range(21):
        on = (dof[:, j] <= lims[j, 0] + 1e-4) | (dof[:, j] >= lims[j, 1] - 1e-4)
        pc = 100.0 * on.sum() / T
        if pc > 0.5:
            fr = frames[np.where(on)[0]]
            req = dof[on, j]
            sat_rows.append((DOF_ORDER[j], pc, int(fr[0]), int(fr[-1]),
                             lims[j, 0], lims[j, 1]))

    # --- largest-error frames ----------------------------------------------------
    per_frame = foot_traj_err.max(1)
    worst = np.argsort(per_frame)[::-1][:8]
    worst_rows = [(int(frames[i]), per_frame[i]*1000,
                   LEGS[int(foot_traj_err[i].argmax())]) for i in worst]

    ear_err = r["ear_orientation_error"].astype(float) if "ear_orientation_error" in r.files else np.zeros((T,2))
    ear_worst_rows = []
    for side, col in (("left", 0), ("right", 1)):
        for i in np.argsort(ear_err[:, col])[::-1][:5]:
            ear_worst_rows.append((side, int(frames[i]), np.degrees(ear_err[i, col])))

    # Mapped source visible-ear body-relative deltas for representative debug rows.
    RearL = np.array([F @ quat_to_mat(q) @ F for q in k["earl_quat"].astype(float)])
    RearR = np.array([F @ quat_to_mat(q) @ F for q in k["earr_quat"].astype(float)])
    Cbody = A @ Rbody_ref[0]
    def ear_deltas(Rsrc):
        H0 = Rbody_ref[0].T @ Rsrc[0]
        return np.array([Cbody @ ((Rbody_ref[i].T @ Rsrc[i]) @ H0.T) @ Cbody.T
                         for i in range(T)])
    ear_delta_l, ear_delta_r = ear_deltas(RearL), ear_deltas(RearR)

    # --- velocities --------------------------------------------------------------
    vel = np.abs(np.gradient(dof, dt, axis=0))
    vleg = vel[:, :12].max(); vexp = vel[:, 12:].max()

    # ------------------------------------------------------------------ report ----
    L = []
    def w(s=""): L.append(s)
    w("# Cheeky -> v4 spatial retarget - evaluation report")
    w()
    w(f"Source: `{str(k['source'])}`  ->  target: exact v4 physical skeleton "
      f"(`Bingo_Cheeky_V4_Retargeted.blend`).  Kinematic only (no physics/RL).")
    w()
    w("## 1. Discovered rig mapping")
    w()
    w("Ashley's rig is a biped-style quadruped (front legs = Arm/ForeArm/Hand, back "
      "legs = Leg/Shin/Foot) driven through `def_*` deform bones. Its anatomical frame "
      "(forward +Y, up +Z) is **mirrored/left-handed** vs the robot (forward +X, left "
      "+Y, up +Z); the source is reflected across YZ before fitting.")
    w()
    w("| Ashley (source) | v4 (target) | role |")
    w("|---|---|---|")
    rolemap = {"fl": "front-left", "fr": "front-right", "bl": "back-left", "br": "back-right"}
    for rl in LEGS:
        w(f"| {RMAP[rl]} ({rolemap[rl]}) | `{rl}_SY_J/SP_J/knee` | leg |")
    w("| def_Pelvis | `root` (floating base) | body |")
    w("| def_Head | `head_pitch_joint/head_yaw/head_roll` | head (3 DOF) |")
    w("| def_Tail.001 | `tail_pitch/tail_yaw` | tail (2 DOF) |")
    w("| Anim_Ear.L / Anim_Ear.R | `l_ear_*` / `r_ear_*` | visible terminal ears (2+2 DOF) |")
    w()
    w(f"Per-leg length scale (source units -> metres): "
      + ", ".join(f"`{rl}`={s_leg[rl]:.5f}" for rl in LEGS) + ".")
    w()
    w("## 2. Source clip")
    w()
    w(f"- Frames **{int(frames[0])}-{int(frames[-1])}** ({T} frames), **{fps:.0f} fps**, "
      f"{T/fps:.2f} s.")
    w(f"- Method: evaluated world-space `def_*` keypoints -> per-leg limb-length "
      f"scaling -> per-frame Kabsch root fit -> contact-anchored per-leg IK "
      f"(scipy `least_squares`, bounds = exact v4 URDF limits) -> head/tail/ear "
      f"rotation-chain solve -> globally continuous absolute contact-anchor root solve -> "
      f"velocity-constrained trajectory refinement.")
    w()
    w("## 3. Contact intervals (source schedule, preserved on the robot)")
    w()

    def ivals(mask):
        out = []; i = 0
        while i < T:
            if mask[i]:
                j = i
                while j < T and mask[j]:
                    j += 1
                out.append(f"{int(frames[i])}-{int(frames[j-1])}"); i = j
            else:
                i += 1
        return ", ".join(out) if out else "(none)"
    w("| foot | duty | planted intervals |")
    w("|---|---|---|")
    for kk, rl in enumerate(LEGS):
        w(f"| {rl} ({rolemap[rl]}) | {100*ct[:,kk].mean():.0f}% | {ivals(ct[:,kk])} |")
    w()
    w(f"Contact schedule preserved (source vs solved, per foot-frame): **{preserved:.1f}%**.")
    w()
    w("## 4. Quantitative errors")
    w()
    w("| metric | mean | p95 | max |")
    w("|---|---|---|---|")
    w(f"| foot trajectory error (hip-relative, scaled) | {foot_traj_err.mean()*1000:.1f} mm "
      f"| {np.percentile(foot_traj_err,95)*1000:.1f} mm | {foot_traj_err.max()*1000:.1f} mm |")
    w(f"| foot IK residual (target vs achieved) | {foot_err.mean()*1000:.1f} mm "
      f"| {np.percentile(foot_err,95)*1000:.1f} mm | {foot_err.max()*1000:.1f} mm |")
    w(f"| knee trajectory error (hip-relative, scaled) | {knee_traj_err.mean()*1000:.1f} mm "
      f"| {np.percentile(knee_traj_err,95)*1000:.1f} mm | {knee_traj_err.max()*1000:.1f} mm |")
    w(f"| root deviation from scaled Ashley body | {root_dev.mean()*1000:.1f} mm "
      f"| {np.percentile(root_dev,95)*1000:.1f} mm | {root_dev.max()*1000:.1f} mm |")
    w(f"| root orientation error | {np.degrees(root_ori_err).mean():.3f} deg "
      f"| {np.percentile(np.degrees(root_ori_err),95):.3f} deg | {np.degrees(root_ori_err).max():.3f} deg |")
    w()
    w(f"- **Planted-foot sliding:** mean {slide_mm:.1f} mm/frame, max {slide_max:.1f} mm "
      f"between consecutive planted frames.")
    w(f"- **Ground penetration:** max {pen_mm:.1f} mm below the floor "
      f"({pen_frames} frame(s) with any paw < -1 mm).")
    w(f"- **Joint velocity:** legs max {vleg:.1f} rad/s (limit 10), expression max "
      f"{vexp:.1f} rad/s (limit 8).")
    w()
    w("### Gait phase evidence")
    w()
    w("| foot | source stance speed | target stance speed | source swing distance | target swing distance |")
    w("|---|---:|---:|---:|---:|")
    for kk, rl in enumerate(LEGS):
        w(f"| {rl} | {stance_speed_src[kk]*1000:.1f} mm/s | {stance_speed_tgt[kk]*1000:.1f} mm/s "
          f"| {swing_dist_src[kk]:.3f} m | {swing_dist_tgt[kk]:.3f} m |")
    w()
    w("## 5. Joint-limit saturation (poses v4 could not fully reproduce)")
    w()
    if sat_rows:
        w("| joint | % frames on limit | frame span | limit (rad) |")
        w("|---|---|---|---|")
        for nm, pc, f0, f1, lo, hi in sat_rows:
            w(f"| `{nm}` | {pc:.1f}% | {f0}-{f1} | [{lo:.2f}, {hi:.2f}] |")
    else:
        w("None above 0.5% of frames.")
    w()
    w("Interpretation: the `SY` (shoulder-yaw/abduction) limit is only "
      "**+/-0.42 rad (~24 deg)**, so Ashley's wide lateral paw placements and the big "
      "front-paw raises are the main gestures v4 physically cannot match one-to-one; "
      "these are recorded here rather than hidden, and the amplitude is **not** globally "
      "shrunk to make the solver succeed.")
    w()
    w("## 6. Root-cause diagnosis and changes")
    w()
    w("- **Gait failure:** leg targets omitted the established source-to-v4 axis matrix `A`; "
      "the evaluator repeated the omission. The old cumulative de-slip was then blurred, so it "
      "neither enforced stance anchors nor preserved source travel. Fixed by applying `A` once, "
      "using absolute per-stance anchors, solving one globally continuous root offset, and "
      "refining joints under explicit velocity bounds instead of globally blurring the clip.")
    w("- **Left-ear failure:** Stage 2 sampled `def_Ear.*`, while the visible meshes inherit "
      "the child `Anim_Ear.*` transforms. It also solved body-relative ear orientation without "
      "removing the already-achieved v4 head rotation. Fixed by sampling visible terminal bones "
      "and solving left/right separately relative to the achieved head and each target rest frame.")
    w("- **Floating bones:** ten `ctrl_*` bones, four `*_ik_end` helpers, and four "
      "`*_foot_tip` markers stayed at the rig rest location while the physical root moved. "
      "All 18 nonphysical bones are removed from the final baked file. Only physical `root` "
      "+ 21 joints remain, are keyed, and are required.")
    w()
    w("## 7. Largest gait-error frames")
    w()
    w("| frame | worst foot | foot error |")
    w("|---|---|---|")
    for fr, e, leg in worst_rows:
        w(f"| {fr} | {leg} | {e:.0f} mm |")
    w()
    w("## 8. Ear mapping and largest ear-error frames")
    w()
    w("The source is reflected exactly once across X (`F R F`) and then aligned with `A`; "
      "ears are not swapped for the selected direct anatomical mapping. Source terminal rest "
      "orientation is removed before mapping. Target axes and asymmetric limits come directly "
      "from the unchanged v4 URDF.")
    w()
    w("| side | target chain | pitch axis | roll axis | limits (rad) |")
    w("|---|---|---|---|---|")
    for side, chain in (("left", ["l_ear_pitch", "l_ear_roll"]),
                        ("right", ["r_ear_pitch", "r_ear_roll"])):
        axes = [np.round(kin.j[n]["axis"],3).tolist() for n in chain]
        limits = [(kin.j[n]["lo"],kin.j[n]["hi"]) for n in chain]
        w(f"| {side} | `{chain[0]}` -> `{chain[1]}` | `{axes[0]}` | `{axes[1]}` | "
          f"[{limits[0][0]:.2f},{limits[0][1]:.2f}], [{limits[1][0]:.2f},{limits[1][1]:.2f}] |")
    w()
    w("| side | frame | terminal orientation error |")
    w("|---|---:|---:|")
    for side, fr, deg in ear_worst_rows:
        w(f"| {side} | {fr} | {deg:.1f} deg |")
    w()
    w("Representative mapped Ashley visible-ear deltas and final v4 joint values (`wxyz`, rad):")
    w()
    w("| frame | Ashley L delta | v4 L pitch/roll | Ashley R delta | v4 R pitch/roll |")
    w("|---:|---|---|---|---|")
    for fr in (1,60,94,121,127,128,180):
        idx = np.where(frames == fr)[0]
        if not len(idx): continue
        i = int(idx[0]); lq=mat_to_quat(ear_delta_l[i]); rq=mat_to_quat(ear_delta_r[i])
        w(f"| {fr} | `{np.round(lq,3).tolist()}` | `{np.round(dof[i,17:19],3).tolist()}` "
          f"| `{np.round(rq,3).tolist()}` | `{np.round(dof[i,19:21],3).tolist()}` |")
    w()
    w("## 9. Assumptions")
    w()
    w("- Paw contact point = tip of Ashley's Hand/Foot deform bone; robot foot = shank "
      f"tip (knee frame + {int(SHANK()*1000)} mm along -Z), paw mesh hangs {int(PAW_DROP*1000)} mm below it.")
    w("- Front/back leg correspondence is fixed by anatomy; left/right is chosen by the "
      "lower hip-layout Kabsch residual (handles the mirrored labels).")
    w("- Expressive chains reproduce Ashley's **in-body orientation delta from frame 1**, "
      "mapped through target rest frames; ears additionally compensate for achieved head pose.")
    w("- Root follows Ashley's body strongly but is allowed to adapt (contact-consistent "
      "de-slip + ground placement) so planted paws stay put and the lowest paw rests on z=0.")
    w()
    w("## 10. Deliverables")
    w()
    for p in ["blend_sources/Bingo_Cheeky_V4_Retargeted.blend",
              "stage2/out/cheeky_source_keypoints.npz",
              "stage2/out/cheeky_contacts.npz",
              "stage2/out/cheeky_v4_retarget.npz",
              "stage2/*.py", "CHEEKY_RETARGET_REPORT.md",
              "stage2/out/cheeky_compare.mp4 (side-by-side)"]:
        w(f"- `{p}`")

    open(a.report, "w").write("\n".join(L) + "\n")
    print(f"[[ foot traj err mean {foot_traj_err.mean()*1000:.1f} mm  slide {slide_mm:.1f} mm  "
          f"contact preserved {preserved:.1f}%  penetration {pen_mm:.1f} mm")
    print(f"[[ wrote {a.report}")


def SHANK():
    from v4_kinematics import SHANK_LEN
    return SHANK_LEN


main()
