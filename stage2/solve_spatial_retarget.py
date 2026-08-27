"""Stage 2.6/7/8/9 - spatial retarget of Ashley's world-space motion onto exact v4.

Method (kinematic only, no physics):
  1. Fix the Ashley->v4 leg correspondence (front/back is anatomy; L/R is chosen
     by whichever mapping gives the lower hip-layout Kabsch residual - handles the
     mirrored rig labels automatically).
  2. Preserve the complete semantic chain SY->SP->knee->ankle->toe.  Ashley's
     ankle/paw-bone HEAD maps to the v4 shank tip; her toe is only a secondary
     target for the v4 rigid paw contact point.
  3. Place the real v4 SY->SP offset exactly once and take the root prior from
     Ashley's pelvis/SP layout.
  4. Solve ankle position + knee shape + orientation-aware collision support.
     During stance the physical support point is locked to its touchdown world
     anchor; during swing the authored ankle/toe trajectory remains active.
  5. Only after the physical leg solve, apply a small smooth root correction and
     re-solve the legs against the same world anchors.
  7. Rate-limit smoothing to respect v4 joint-velocity limits. Limit saturation and
     large deviations are RECORDED, never hidden.

Run (system python, needs scipy):
  python3 stage2/solve_spatial_retarget.py \
      --keypoints stage2/out/cheeky_source_keypoints.npz \
      --contacts  stage2/out/cheeky_contacts.npz \
      --urdf "URDF/bingo_urdf v4_w_ear_joints/urdf/bingo_urdf_w_ear_joints.urdf" \
      --out stage2/out/cheeky_v4_retarget.npz
"""
import argparse, sys, os
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stage4")))
from v4_kinematics import (V4Kin, LEGS, DOF_ORDER, HEAD_CHAIN, TAIL_CHAIN, axis_rot,
                           LEAR_CHAIN, REAR_CHAIN, mat_to_quat, quat_to_mat,
                           rotvec_of)
from contact_model import ContactModel, HULLS_NPZ

ALEGS = ["aFL", "aFR", "aBL", "aBR"]
# two candidate L/R correspondences (front/back is fixed by anatomy)
MAP_DIRECT = {"aFL": "fl", "aFR": "fr", "aBL": "bl", "aBR": "br"}
MAP_MIRROR = {"aFL": "fr", "aFR": "fl", "aBL": "br", "aBR": "bl"}


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def gsmooth(x, sig):
    if sig <= 0:
        return x
    r = int(np.ceil(3 * sig))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); w /= w.sum()
    pad = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, c], w, "valid") for c in range(x.shape[1])], 1)


def solve_leg(kin, leg, ankle_target, knee_target, contact_target, q0, lim,
              support_hull, root_R, root_pos=None, contact_weight=2.0,
              continuity=0.0, floor_z=None, contact_world_target=None):
    """Solve one physical v4 leg without inventing an ankle DOF.

    Position weights follow the requested priority 5:3:2. ``continuity`` is a
    metres-per-radian scale so the temporal term remains comparable to position.
    """
    def resid(q):
        _, knee, ankle, contact, _ = kin.leg_points(
            leg, q, support_hull=support_hull, world_R=root_R,
            support_softness=0.001)
        if contact_world_target is None:
            contact_residual = contact - contact_target
        else:
            active = kin.leg_points(leg, q, support_hull=support_hull,
                                    world_R=root_R)[3]
            world_patch = root_pos + root_R @ contact
            world_low_z = float(root_pos[2] + (root_R @ active)[2])
            # A grounded rigid paw has three support constraints: collision-patch
            # XY at its touchdown anchor and exact lowest-hull height at ground Z.
            contact_residual = np.array([
                world_patch[0] - contact_world_target[0],
                world_patch[1] - contact_world_target[1],
                world_low_z - contact_world_target[2],
            ])
        pos = np.r_[np.sqrt(5.0) * (ankle - ankle_target),
                    np.sqrt(3.0) * (knee - knee_target),
                    np.sqrt(contact_weight) * contact_residual]
        # Swing is free to follow Ashley, but not to put collision geometry below
        # the floor. This is a one-sided physical constraint, not root-Z grounding.
        if floor_z is not None and root_pos is not None and contact_world_target is None:
            active = kin.leg_points(leg, q, support_hull=support_hull,
                                    world_R=root_R)[3]
            wz = float(root_pos[2] + (root_R @ active)[2])
            pos = np.r_[pos, np.sqrt(1000.0) * min(0.0, wz - floor_z)]
        if continuity > 0:
            return np.r_[pos, continuity * (q - q0)]
        return pos
    r = least_squares(resid, q0, bounds=(lim[:, 0], lim[:, 1]),
                      method='trf', xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=200)
    _, knee, ankle, contact, _ = kin.leg_points(
        leg, r.x, support_hull=support_hull, world_R=root_R,
        support_softness=0.001)
    if contact_world_target is None:
        cerr = float(np.linalg.norm(contact - contact_target))
    else:
        active = kin.leg_points(leg, r.x, support_hull=support_hull,
                                world_R=root_R)[3]
        wp = root_pos + root_R @ contact
        wz = float(root_pos[2] + (root_R @ active)[2])
        cerr = float(np.linalg.norm(np.r_[wp[:2] - contact_world_target[:2],
                                         wz - contact_world_target[2]]))
    errs = (float(np.linalg.norm(ankle - ankle_target)),
            float(np.linalg.norm(knee - knee_target)),
            cerr)
    return r.x, errs


def solve_chain(kin, names, R_target, q0, lim, continuity=0.0, q_prev=None):
    """Orientation IK for one expressive chain, seeded from the previous frame.

    `continuity` matters wherever the target is OUT OF REACH, which for the head is
    most of several clips: a 3-DOF chain against an unreachable orientation has
    several almost-equally-bad minima, and a purely local solve will hop between them
    from one frame to the next. On Eccentric that produced a 22.9 deg single-frame
    FLICK of the head at robot frames 236-259 - the head sat pinned at head_roll's
    +44.7 deg stop, snapped across the workspace and snapped back. The eye reads that
    as a glitch, not as a joint limit. A small penalty toward the previous frame's
    solution costs almost nothing where the target IS reachable (the orientation
    residual dominates) and picks the near branch where it is not."""
    qp = q0 if q_prev is None else q_prev

    def resid(q):
        r = rotvec_of(kin.chain_rot(names, q).T @ R_target)
        if continuity > 0.0:
            return np.concatenate([r, continuity * (q - qp)])
        return r
    r = least_squares(resid, q0, bounds=(lim[:, 0], lim[:, 1]),
                      method='trf', xtol=1e-12, ftol=1e-12, max_nfev=200)
    err = float(np.linalg.norm(rotvec_of(kin.chain_rot(names, r.x).T @ R_target)))
    return r.x, err


LEG_SEEDS = [np.array([0.0, -0.30, 0.60]), np.array([0.0, 0.30, -0.60]),
             np.array([0.0, -0.30, -0.60]), np.array([0.0, 0.30, 0.60])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoints", required=True)
    ap.add_argument("--contacts", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--collision-hulls", default=HULLS_NPZ)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root-cap", type=float, default=0.035,
                    help="metres. Bound on the residual body offset the contact solve "
                         "may add on top of Ashley's scaled hip-centroid path. This is a "
                         "direct trade: a tighter cap keeps the body exactly where Ashley "
                         "put it and forces the planted paws to skate instead; a looser "
                         "one moves the body a little and keeps the feet welded to the "
                         "floor. Physics only tracks the second kind, so prefer to loosen "
                         "it and MEASURE the root deviation rather than pay in slip.")
    ap.add_argument("--expr-vel", type=float, default=8.0,
                    help="rad/s cap on the HEAD and TAIL joints, applied "
                         "as a forward-backward projection after the chain solve. 8.0 "
                         "is the v4 head/tail velocity limit. Lowering it is the "
                         "cheaper way to stop an out-of-reach head FLICKING between "
                         "equivalent minima - it bounds how fast the chain may cross "
                         "the workspace without penalising the reachable part of the "
                         "motion the way --expr-continuity does.")
    ap.add_argument("--yaw-align", choices=("heading", "travel"), default="heading",
                    help="which direction the exported world frame is built around: "
                         "the character's frame-0 heading (default, unchanged) or its "
                         "net travel. Cosmetic - a rigid yaw about Z - but it decides "
                         "whether a crabbing clip runs along a Blender axis or across "
                         "the grid diagonally.")
    ap.add_argument("--ear-vel", type=float, default=8.0,
                    help="rad/s cap for the EAR joints only, kept separate from "
                         "--expr-vel so slowing a limit-bound head does not also slow "
                         "the ears, which track Ashley well on every clip.")
    ap.add_argument("--expr-continuity", type=float, default=0.15,
                    help="temporal continuity weight for the head/tail/ear orientation "
                         "IK, in radians of joint change per radian of orientation "
                         "error. Stops the chain hopping between equivalent minima "
                         "when the target is past the v4 limits (see solve_chain).")
    ap.add_argument("--stance-weight", type=float, default=2500.0,
                    help="how hard a scheduled stance foot is held to its world "
                         "touchdown anchor, against the 5:3:2 ankle:knee:toe objective "
                         "that shapes the leg. High keeps the foot welded to the floor "
                         "and lets the authored ankle position drift; low reproduces "
                         "Ashley's leg more exactly and lets the foot skate. Physics "
                         "only tracks the first kind - measure before lowering it.")
    ap.add_argument("--root-step", type=float, default=0.004,
                    help="metres/frame rate limit on that offset (keeps it smooth)")
    a = ap.parse_args()

    d = np.load(a.keypoints, allow_pickle=True)
    c = np.load(a.contacts, allow_pickle=True)
    kin = V4Kin(a.urdf)
    contact_model = ContactModel(a.collision_hulls)
    support_hull = {l: contact_model.hull[f"{l}_knee"] for l in LEGS}
    fps = float(d["fps"]); frames = d["frames"]; T = len(frames); dt = 1.0 / fps

    # Ashley's rig is MIRRORED: its anatomical frame (forward=+Y, left=+X, up=+Z)
    # is LEFT-handed relative to the robot (forward=+X, left=+Y, up=+Z). A proper
    # rotation cannot map a left-handed point cloud onto a right-handed one, so
    # Kabsch would place the feet wrong (the tell: direct & mirror residuals equal).
    # Reflect the source across the YZ plane (negate X) once at load -> right-handed.
    # Positions: p -> F p.  Rotations: R -> F R F (stays a proper rotation).
    F = np.diag([-1.0, 1.0, 1.0])
    rf = lambda P: P @ F                                # (T,3) @ diag = negate col 0
    rfR = lambda R: F @ R @ F

    sy = {l: rf(d[f"sy_{l}"].astype(float)) for l in ALEGS}
    sp = {l: rf(d[f"sp_{l}"].astype(float)) for l in ALEGS}
    knee = {l: rf(d[f"knee_{l}"].astype(float)) for l in ALEGS}
    ankle = {l: rf(d[f"ankle_{l}"].astype(float)) for l in ALEGS}
    toe = {l: rf(d[f"toe_{l}"].astype(float)) for l in ALEGS}
    body_pos = rf(d["body_pos"].astype(float))
    Rbody = np.array([rfR(quat_to_mat(q)) for q in d["body_quat"].astype(float)])
    Rhead = np.array([rfR(quat_to_mat(q)) for q in d["head_quat"].astype(float)])
    Rtail = np.array([rfR(quat_to_mat(q)) for q in d["tail_quat"].astype(float)])
    RearL = np.array([rfR(quat_to_mat(q)) for q in d["earl_quat"].astype(float)])
    RearR = np.array([rfR(quat_to_mat(q)) for q in d["earr_quat"].astype(float)])
    rest = {l: d["rest_lengths"][i] for i, l in enumerate(ALEGS)}
    contacts_a = c["contacts"]                          # (T,4) in ALEGS order

    # ---- per-leg scale (Ashley limb length -> v4 reach), by DIRECT map first ----
    def scale_for(mp):
        s = {}
        for al in ALEGS:
            rl = mp[al]
            s[rl] = kin.leg_reach(rl) / float(rest[al].sum())
        return s

    # ---- choose L/R correspondence by hip-layout Kabsch residual ---------------
    urdf_sy = {l: kin.sy_pos(l) for l in LEGS}
    urdf_sp = {l: kin.sp_pos_zero(l) for l in LEGS}
    probe = np.linspace(0, T - 1, min(30, T), dtype=int)

    def map_residual(mp):
        s = float(np.mean([kin.leg_reach(mp[al]) / float(rest[al].sum()) for al in ALEGS]))
        res = []
        for i in probe:
            # Use both semantic pivots.  The old SP->SY comparison hid the
            # missing target shoulder offset that caused the folded-leg result.
            P = np.array([p for al in ALEGS for p in (sy[al][i], sp[al][i])]) * s
            Q = np.array([p for al in ALEGS for p in
                          (urdf_sy[mp[al]], urdf_sp[mp[al]])])
            R, t = kabsch(P, Q)
            res.append(np.sqrt(np.mean(np.sum((P @ R.T + t - Q) ** 2, 1))))
        return float(np.mean(res))

    rd, rm = map_residual(MAP_DIRECT), map_residual(MAP_MIRROR)
    # The bilateral layout is symmetric, so the two residuals can be equal to
    # floating-point noise.  Preserve the established anatomical mapping on a
    # tie; only mirror when it is materially (>1 mm) better.
    MAP = MAP_MIRROR if rm + 0.001 < rd else MAP_DIRECT
    print(f"[[ leg map: direct residual {rd*1000:.1f} mm | mirror {rm*1000:.1f} mm "
          f"-> using {'DIRECT' if MAP is MAP_DIRECT else 'MIRROR'}")
    print(f"[[ correspondence: " + "  ".join(f"{al}->{MAP[al]}" for al in ALEGS))
    # keep ear side consistent with the chosen leg side
    if MAP is MAP_MIRROR:
        RearL, RearR = RearR, RearL
    # invert: robot leg -> ashley leg
    RMAP = {v: k for k, v in MAP.items()}
    s_leg = scale_for(MAP)
    s_master = float(np.mean(list(s_leg.values())))
    print(f"[[ per-leg scale (m/unit): "
          + "  ".join(f"{l}={s_leg[l]:.5f}" for l in LEGS) + f"  master={s_master:.5f}")

    # ---- per-frame root pose --------------------------------------------------
    # The 4 hips are nearly COPLANAR (all at ~one height), so Kabsch on them is
    # degenerate about the vertical: it can flip the up-axis while fitting the flat
    # hip rectangle equally well, sending the feet the wrong way. Take the root
    # ORIENTATION from Ashley's pelvis (a full non-degenerate rigid frame) instead,
    # and only use the hip geometry once to establish the source->robot axis
    # alignment A. Position comes from the (well-conditioned) hip centroid.
    def cen(idx0):  # frame-0 centroid of Ashley's SP pivots
        return np.mean([sp[al][0] for al in idx0], 0)
    front_c = cen([RMAP["fl"], RMAP["fr"]]); back_c = cen([RMAP["bl"], RMAP["br"]])
    left_c = cen([RMAP["fl"], RMAP["bl"]]); right_c = cen([RMAP["fr"], RMAP["br"]])
    # A must be a pure YAW alignment. Deriving "up" from the frame-0 hip plane
    # assumes the character starts level; when a clip opens lying down, rearing or
    # crouched (DeadPan/Eccentric/Enthusiastic start 38-42 deg off level) that
    # normal is NOT world up, and A then tilts the whole trajectory - turning the
    # source's horizontal travel into robot vertical travel, i.e. the robot flies
    # and skates in mid-air. Both frames are +Z up (the reflection already fixed
    # handedness), so take up = world +Z and use the hip geometry only for heading.
    up0 = np.array([0.0, 0.0, 1.0])
    fwd0 = front_c - back_c
    fwd0 = fwd0 - up0 * (fwd0 @ up0)                    # project heading to horizontal
    n = np.linalg.norm(fwd0)
    if n < 1e-6:                                        # degenerate: fall back to L/R
        fwd0 = np.cross(left_c - right_c, up0); n = np.linalg.norm(fwd0)
    fwd0 /= n
    lft0 = np.cross(up0, fwd0)
    S0 = np.column_stack([fwd0, lft0, up0])            # source-world axes at rest
    A = S0.T                                            # source-world -> robot-origin
    print(f"[[ axis align: yaw-only, heading {np.degrees(np.arctan2(fwd0[1], fwd0[0])):+.1f} deg "
          f"(up forced to world +Z)")
    if a.yaw_align == "travel":
        # Put the clip's NET TRAVEL along world +X. Purely a choice of world frame -
        # a rigid yaw about Z, with gravity along Z, so nothing physical changes. It
        # exists because the default (frame-0 heading -> +X) leaves a clip whose
        # character CRABS running diagonally across the Blender grid, which reads as
        # a retarget error when it is not. Eccentric is the case: Ashley's body faces
        # 27.0 deg off her own direction of travel and the v4 retarget reproduces
        # that to 0.2 deg, but her travel happens to lie along her -Y axis while ours
        # ends up 28 deg off +X.
        _c = np.array([np.mean([sp[RMAP[l]][i] for l in LEGS], 0) for i in (0, T - 1)])
        _t = (A @ (_c[1] - _c[0]))[:2]
        if np.linalg.norm(_t) > 1e-9:
            _t /= np.linalg.norm(_t)
            _R = np.array([[_t[0], _t[1], 0.0], [-_t[1], _t[0], 0.0], [0.0, 0.0, 1.0]])
            A = _R @ A
            print(f"[[ yaw-align=travel: rotated the world frame by "
                  f"{np.degrees(np.arctan2(-_t[1], _t[0])):+.1f} deg so the net travel "
                  f"runs along +X")
    meansp = np.mean([urdf_sp[l] for l in LEGS], 0)
    sp_centroid = np.array([np.mean([sp[RMAP[l]][i] for l in LEGS], 0)
                            for i in range(T)])

    # Root orientation must be ABSOLUTE, not a delta from frame 0. A delta silently
    # declares frame 0 to be level; DeadPan opens 41 deg tilted and Eccentric 47 deg
    # (it is a reclining performance), so the whole clip inherited that constant
    # error and the robot ended up tipped over / hanging in the air even with the
    # feet numerically at z=0. Take Ashley's per-frame ANATOMICAL frame from her
    # body-attached SY pivots, and map it through the yaw-only alignment A.
    def anat_frame(i):
        fc = 0.5 * (sy[RMAP["fl"]][i] + sy[RMAP["fr"]][i])
        bc = 0.5 * (sy[RMAP["bl"]][i] + sy[RMAP["br"]][i])
        lc = 0.5 * (sy[RMAP["fl"]][i] + sy[RMAP["bl"]][i])
        rc = 0.5 * (sy[RMAP["fr"]][i] + sy[RMAP["br"]][i])
        f = fc - bc; f /= np.linalg.norm(f) + 1e-12
        u = np.cross(f, lc - rc); u /= np.linalg.norm(u) + 1e-12
        return np.column_stack([f, np.cross(u, f), u])
    S_anat = np.array([anat_frame(i) for i in range(T)])

    # The pelvis quaternion is smoother than the hip-plane estimate, so drive the
    # orientation with it but CALIBRATE it against anatomy over the whole clip
    # (not against frame 0). C maps Ashley's pelvis-local axes to anatomical axes.
    Cm = np.mean([Rbody[i].T @ S_anat[i] for i in range(T)], axis=0)
    U_, _, Vt_ = np.linalg.svd(Cm)                       # nearest proper rotation
    C_anat = U_ @ np.diag([1, 1, np.sign(np.linalg.det(U_ @ Vt_))]) @ Vt_

    R_b = np.zeros((T, 3, 3)); t_b = np.zeros((T, 3))
    for i in range(T):
        R_b[i] = A @ Rbody[i] @ C_anat                  # absolute origin orientation
        spc = sp_centroid[i] * s_master
        t_b[i] = A @ spc - R_b[i] @ meansp              # target SP centroid follows source
    tilt = np.degrees(np.arccos(np.clip(R_b[:, 2, 2], -1, 1)))
    print(f"[[ body tilt from level (absolute): mean {tilt.mean():.1f} deg  "
          f"f0 {tilt[0]:.1f}  min {tilt.min():.1f}  max {tilt.max():.1f}")
    # report the unavoidable rest-layout mismatch separately from animation
    fitres = np.zeros(T)
    for i in range(T):
        err = []
        for l in LEGS:
            w = t_b[i] + R_b[i] @ urdf_sp[l]
            src = A @ (sp[RMAP[l]][i] * s_master)
            err.append(np.linalg.norm(w - src))
        fitres[i] = np.mean(err)
    print(f"[[ root: orientation from pelvis, position from hip centroid | hip-place "
          f"residual mean {fitres.mean()*1000:.1f} mm max {fitres.max()*1000:.1f} mm")

    # ---- SP-relative semantic leg targets --------------------------------------
    ankle_tgt = np.zeros((T, 4, 3))
    knee_tgt = np.zeros((T, 4, 3))
    contact_tgt = np.zeros((T, 4, 3))
    world_contact_source = np.zeros((T, 4, 3))
    sp0_centroid = sp_centroid[0]
    for i in range(T):
        for k, l in enumerate(LEGS):
            al = RMAP[l]
            # Carry only Ashley's animated SP displacement onto the exact v4
            # rest layout; do not force the v4 pivots into Ashley's narrower rig.
            #
            # This delta MUST be taken in Ashley's own body frame. Differencing
            # the WORLD-space hip offset against frame 0 (the old form) measures
            # the body's rigid ROTATION as if it were scapula translation: a hip
            # 70 mm off the centroid on a clip that yaws 90 deg reads as ~100 mm
            # of shoulder travel that never happened. Measured on the six clips,
            # the world-frame form reports 19-79 mm mean (up to 194 mm) of hip
            # motion and tracks body yaw almost perfectly, while the true
            # body-frame motion is 1.5-9.3 mm mean, <=25 mm max. Added to a leg
            # whose ankle target already sits at 150-200 mm of a 203.6 mm reach,
            # that artefact pushed DeadPan's front-leg targets outside the
            # reachable ball on 78% of frames (mean ankle IK error 62 mm) and
            # forced the planted paws to skate. Mapping both endpoints through
            # R_b[i].T @ A cancels the rotation exactly, since
            # R_b[i].T @ A == C_anat.T @ Rbody[i].T.
            sp_off_i = R_b[i].T @ (A @ (sp[al][i] - sp_centroid[i]))
            sp_off_0 = R_b[0].T @ (A @ (sp[al][0] - sp0_centroid))
            sp_target = urdf_sp[l] + (sp_off_i - sp_off_0) * s_master

            avec = R_b[i].T @ (A @ ((ankle[al][i] - sp[al][i]) * s_leg[l]))
            ankle_tgt[i, k] = sp_target + avec

            # Match knee direction/plane using the target's own upper-leg
            # length, avoiding a false penalty from the small morphology ratio.
            kdir = R_b[i].T @ (A @ (knee[al][i] - sp[al][i]))
            kdir /= np.linalg.norm(kdir) + 1e-12
            knee_tgt[i, k] = sp_target + kdir * kin.upper_len(l)

            cvec = R_b[i].T @ (A @ ((toe[al][i] - sp[al][i]) * s_leg[l]))
            contact_tgt[i, k] = sp_target + cvec
            world_contact_source[i, k] = t_b[i] + R_b[i] @ contact_tgt[i, k]

    # Source contact schedule in robot-leg order. It defines stance/swing timing;
    # the actual exported contacts are recomputed from collision support later.
    ct = np.zeros((T, 4), bool)
    for k, l in enumerate(LEGS):
        ct[:, k] = contacts_a[:, ALEGS.index(RMAP[l])]

    # ---- preliminary morphology solve -----------------------------------------
    # First solve Ashley's ankle/knee/toe shape with the real support hull but no
    # world lock. The achieved physical point at each touchdown, rather than an
    # averaged source toe or a fixed local point, establishes the stance anchor.
    dof = np.zeros((T, 21))
    ankle_err = np.zeros((T, 4)); knee_err = np.zeros((T, 4)); contact_err = np.zeros((T, 4))
    lims_leg = {l: kin.leg_limits(l) for l in LEGS}
    q_prev = {}
    for k, l in enumerate(LEGS):
        best = None
        for sd in LEG_SEEDS:
            q, errs = solve_leg(kin, l, ankle_tgt[0, k], knee_tgt[0, k],
                                contact_tgt[0, k], sd, lims_leg[l],
                                support_hull[l], R_b[0], contact_weight=2.0)
            score = 5*errs[0]**2 + 3*errs[1]**2 + 2*errs[2]**2
            if best is None or score < best[1]:
                best = (q, score)
        q_prev[l] = best[0]
    for i in range(T):
        for k, l in enumerate(LEGS):
            q, errs = solve_leg(kin, l, ankle_tgt[i, k], knee_tgt[i, k],
                                contact_tgt[i, k], q_prev[l], lims_leg[l],
                                support_hull[l], R_b[i], contact_weight=2.0,
                                continuity=0.008)
            dof[i, 3*k:3*k+3] = q
            ankle_err[i, k], knee_err[i, k], contact_err[i, k] = errs
            q_prev[l] = q

    # Preserve the morphology solve's knee branch. Foot position alone admits a
    # reflected/folded solution; Timid's correct branch is unambiguous here.
    knee_branch = {l: np.sign(np.median(dof[:, 3*k+2])) or 1.0
                   for k, l in enumerate(LEGS)}

    def support_origin(i, k, qleg):
        l = LEGS[k]
        return kin.leg_points(l, qleg, support_hull=support_hull[l],
                              world_R=R_b[i], support_softness=0.001)[3]

    initial_support = np.zeros((T, 4, 3))
    initial_lowest = np.zeros((T, 4, 3))
    for i in range(T):
        for k in range(4):
            cp = support_origin(i, k, dof[i, 3*k:3*k+3])
            initial_support[i, k] = t_b[i] + R_b[i] @ cp
            l = LEGS[k]
            lp = kin.leg_points(l, dof[i, 3*k:3*k+3],
                                support_hull=support_hull[l], world_R=R_b[i])[3]
            initial_lowest[i, k] = t_b[i] + R_b[i] @ lp

    # Put the clip onto one physical floor with one clip-wide shift. This replaces
    # the old independent per-frame root-Z grounding that broke temporal locks.
    _child = {}
    for _n, _j in kin.j.items():
        _child.setdefault(_j["parent"], []).append(_n)

    def whole_body_low(i, want_link=False):
        """Lowest collision-hull z over EVERY link at frame i, in world."""
        dd = {DOF_ORDER[c]: dof[i, c] for c in range(21)}
        fr = {"origin": (R_b[i], t_b[i].copy())}
        st = ["origin"]; best = 1e9
        while st:
            par = st.pop(); Rp, pp = fr[par]
            for jn in _child.get(par, []):
                J = kin.j[jn]; pj = pp + Rp @ J["xyz"]
                fr[J["child"]] = (Rp @ J["R"] @ axis_rot(J["axis"], dd.get(jn, 0.0)), pj)
                st.append(J["child"])
        who = None
        for ln, (R_, p_) in fr.items():
            if ln in contact_model.hull:
                z_ = float((contact_model.hull[ln] @ R_.T + p_)[:, 2].min())
                if z_ < best:
                    best, who = z_, ln
        return (best, who) if want_link else best

    touchdown = []
    for k in range(4):
        touchdown.extend(i for i in range(T) if ct[i, k] and (i == 0 or not ct[i-1, k]))
    if not touchdown:
        raise RuntimeError("source contact schedule contains no touchdown")
    # The floor is the plane that the robot RESTS on, which is not always a paw.
    # Eccentric is an authored SIT: only the two front paws ever touch down, and
    # levelling on them alone left the torso hull 40-46 mm BELOW the floor for all
    # 370 frames. Take the whole-robot lowest point at each scheduled stance frame
    # instead - for a walk that point IS the planted paw, so nothing changes, and
    # for a sit it is the torso, which is exactly the surface being sat on.
    stance_frames = np.where(ct.any(1))[0]
    low_link = [whole_body_low(i, True) for i in stance_frames]
    torso_frac = float(np.mean([not any(w[1].startswith(l + "_") for l in LEGS)
                                for w in low_link])) if low_link else 0.0
    touchdown_z = [initial_lowest[i, k, 2] for k in range(4) for i in range(T)
                   if ct[i, k] and (i == 0 or not ct[i-1, k])]
    paw_level = float(np.median(touchdown_z))
    body_level = float(np.median([w[0] for w in low_link])) if low_link else paw_level
    # Which surface is the clip resting ON? If the lowest point during stance is a
    # LEG link, it is a walk and the touchdown paws define the floor. If it is the
    # torso/head/tail, it is a SIT and the paws never define it - Eccentric levelled
    # on its two front touchdowns left the torso hull 40-46 mm underground for all
    # 370 frames. Measured, forcing the whole-body rule on the walking clips instead
    # regresses them (DeadPan static support 93% -> 73%, slip 1.19 -> 1.49 mm/frame),
    # because their lowest stance point is a thigh that legitimately hangs below the
    # paw. So switch on WHICH LINK, not on the height.
    use_body = torso_frac > 0.5
    ground_level = body_level if use_body else paw_level
    print(f"[[ floor: {'whole-body (SIT: torso is lowest on '+format(100*torso_frac,'.0f')+'% of stance frames)' if use_body else 'paw touchdowns (walk)'}"
          f" -> {ground_level*1000:+.1f} mm | paw rule {paw_level*1000:+.1f} mm, "
          f"whole-body rule {body_level*1000:+.1f} mm")
    t_b[:, 2] -= ground_level
    initial_support[:, :, 2] -= ground_level
    initial_lowest[:, :, 2] -= ground_level
    world_contact_source[:, :, 2] -= ground_level

    # Freeze the achieved physical collision support at TOUCHDOWN, projected to
    # the common ground plane.  The same anchor is used for every frame in that
    # contiguous stance run.
    authored_contact_tgt = contact_tgt.copy()
    contact_anchor = np.full((T, 4, 3), np.nan)
    for k in range(4):
        i = 0
        while i < T:
            if not ct[i, k]:
                i += 1
                continue
            j = i + 1
            while j < T and ct[j, k]:
                j += 1
            anchor = initial_support[i, k].copy()
            anchor[2] = 0.0
            contact_anchor[i:j, k] = anchor
            i = j

    def refresh_contact_targets(root_pos):
        contact_tgt[:] = authored_contact_tgt
        for i in range(T):
            for k in np.where(ct[i])[0]:
                contact_tgt[i, k] = R_b[i].T @ (contact_anchor[i, k] - root_pos[i])

    refresh_contact_targets(t_b)

    def solve_leg_frame(i, k, q0, bounds, continuity):
        """Solve one frame using the smooth collision-patch support location."""
        l = LEGS[k]
        bounds = np.asarray(bounds, dtype=float).copy()
        if knee_branch[l] > 0:
            bounds[2, 0] = max(bounds[2, 0], 0.0)
        else:
            bounds[2, 1] = min(bounds[2, 1], 0.0)
        q0 = np.minimum(np.maximum(np.asarray(q0), bounds[:, 0] + 1e-10),
                        bounds[:, 1] - 1e-10)
        if not ct[i, k]:
            return solve_leg(kin, l, ankle_tgt[i, k], knee_tgt[i, k],
                             contact_tgt[i, k], q0, bounds, support_hull[l], R_b[i],
                             root_pos=t_b[i], contact_weight=2.0,
                             continuity=continuity, floor_z=0.0)
        return solve_leg(kin, l, ankle_tgt[i, k], knee_tgt[i, k],
                         contact_tgt[i, k], q0, bounds, support_hull[l], R_b[i],
                         root_pos=t_b[i], contact_weight=a.stance_weight,
                         continuity=continuity, floor_z=0.0,
                         contact_world_target=contact_anchor[i, k])

    # ---- stance-locked physical leg solve -------------------------------------
    # Contact dominates only during stance. Swing retains the requested 5:3:2
    # ankle:knee:toe objective, so expressive leg motion is not flattened.
    q_prev = {l: dof[0, 3*k:3*k+3].copy() for k, l in enumerate(LEGS)}
    for i in range(T):
        for k, l in enumerate(LEGS):
            q0 = dof[i, 3*k:3*k+3] if i == 0 else q_prev[l]
            q, errs = solve_leg_frame(i, k, q0, lims_leg[l], continuity=0.012)
            dof[i, 3*k:3*k+3] = q
            ankle_err[i, k], knee_err[i, k], contact_err[i, k] = errs
            q_prev[l] = q
    print(f"[[ physical floor: preliminary touchdown median {ground_level*1000:+.1f} mm "
          f"-> one clip-wide shift; {int(ct.sum())} scheduled stance foot-frames")

    # ---- head / tail / ears: reproduce Ashley's in-body orientation delta -------
    # In-body delta Hb_i Hb_0^T lives in source-body coords; conjugate into robot
    # origin coords with C = A @ Rbody[0] (source-body -> source-world -> robot-origin).
    C = A @ Rbody[0]
    def chain_targets(Rsrc):
        Hb0 = Rbody[0].T @ Rsrc[0]
        out = np.zeros((T, 3, 3))
        for i in range(T):
            Hb = Rbody[i].T @ Rsrc[i]
            dH = C @ (Hb @ Hb0.T) @ C.T
            out[i] = dH                       # times chain_rot(0), applied in solve
        return out
    hd_t = chain_targets(Rhead); tl_t = chain_targets(Rtail)
    el_t = chain_targets(RearL); er_t = chain_targets(RearR)
    R0 = {n: kin.chain_rot(c, np.zeros(len(c)))
          for n, c in (("h", HEAD_CHAIN), ("t", TAIL_CHAIN),
                       ("l", LEAR_CHAIN), ("r", REAR_CHAIN))}
    lim_h = kin.chain_limits(HEAD_CHAIN); lim_t = kin.chain_limits(TAIL_CHAIN)
    lim_l = kin.chain_limits(LEAR_CHAIN); lim_r = kin.chain_limits(REAR_CHAIN)
    qh = np.zeros(3); qt = np.zeros(2); ql = np.zeros(2); qr = np.zeros(2)
    qh_free = np.zeros(3); qt_free = np.zeros(2)
    ql_free = np.zeros(2); qr_free = np.zeros(2)
    WIDE = 3.5                       # radians of headroom for the unconstrained solve
    wide_h = np.column_stack([np.full(3, -WIDE), np.full(3, WIDE)])
    wide_t = np.column_stack([np.full(2, -WIDE), np.full(2, WIDE)])
    wide_l = wide_r = np.column_stack([np.full(2, -WIDE), np.full(2, WIDE)])
    demand = np.zeros((T, 9))
    ori_lim = np.zeros((T, 4)); ori_free = np.zeros((T, 4))
    hres = np.zeros(T); ear_res = np.zeros((T, 2))
    for i in range(T):
        qh, eh = solve_chain(kin, HEAD_CHAIN, hd_t[i] @ R0["h"], qh, lim_h,
                             a.expr_continuity, qh)
        qt, et = solve_chain(kin, TAIL_CHAIN, tl_t[i] @ R0["t"], qt, lim_t,
                             a.expr_continuity, qt)
        # The ears inherit the achieved physical head.  Solve each visible
        # terminal ear orientation relative to that head, using the complete
        # source-body rest->animated delta and the target's own zero frames.
        H = kin.chain_rot(HEAD_CHAIN, qh)
        target_l_full = el_t[i] @ (R0["h"] @ R0["l"])
        target_r_full = er_t[i] @ (R0["h"] @ R0["r"])
        ql, _ = solve_chain(kin, LEAR_CHAIN, H.T @ target_l_full, ql, lim_l,
                            a.expr_continuity, ql)
        qr, _ = solve_chain(kin, REAR_CHAIN, H.T @ target_r_full, qr, lim_r,
                            a.expr_continuity, qr)
        dof[i, 12:15] = qh; dof[i, 15:17] = qt; dof[i, 17:19] = ql; dof[i, 19:21] = qr
        # What the SOURCE actually asks of each expressive joint, with the v4 limits
        # lifted. The clipped angle above says the joint is at its stop; this says by
        # how much Ashley overshot it, which is the number the mechanical team needs.
        qh_f, _ = solve_chain(kin, HEAD_CHAIN, hd_t[i] @ R0["h"], qh_free, wide_h)
        qt_f, _ = solve_chain(kin, TAIL_CHAIN, tl_t[i] @ R0["t"], qt_free, wide_t)
        # The ears are solved against the ACHIEVED (limited) head in both cases, so
        # the comparison isolates the ear's own limits instead of also inheriting the
        # head's clipping.
        ql_f, _ = solve_chain(kin, LEAR_CHAIN, H.T @ target_l_full, ql_free, wide_l)
        qr_f, _ = solve_chain(kin, REAR_CHAIN, H.T @ target_r_full, qr_free, wide_r)
        demand[i, 0:3] = qh_f; demand[i, 3:5] = qt_f
        demand[i, 5:7] = ql_f; demand[i, 7:9] = qr_f
        qh_free, qt_free, ql_free, qr_free = qh_f, qt_f, ql_f, qr_f
        # Orientation error of the LIMITED solve against the target, and of the
        # unlimited one. The difference is the part the JOINT LIMITS cost, as opposed
        # to the part the 3-DOF chain cannot represent at any limit. Unlike a joint
        # angle this is unambiguous - a pitch-yaw-roll chain has several equivalent
        # decompositions, which is why the raw demand can read 166 deg.
        for _ci, (_ch, _qa, _qb, _tg) in enumerate((
                (HEAD_CHAIN, qh, qh_f, hd_t[i] @ R0["h"]),
                (TAIL_CHAIN, qt, qt_f, tl_t[i] @ R0["t"]),
                (LEAR_CHAIN, ql, ql_f, H.T @ target_l_full),
                (REAR_CHAIN, qr, qr_f, H.T @ target_r_full))):
            ori_lim[i, _ci] = np.linalg.norm(rotvec_of(
                kin.chain_rot(_ch, _qa).T @ _tg))
            ori_free[i, _ci] = np.linalg.norm(rotvec_of(
                kin.chain_rot(_ch, _qb).T @ _tg))
        hres[i] = eh
        ear_res[i, 0] = np.linalg.norm(rotvec_of((H @ kin.chain_rot(LEAR_CHAIN, ql)).T
                                                 @ target_l_full))
        ear_res[i, 1] = np.linalg.norm(rotvec_of((H @ kin.chain_rot(REAR_CHAIN, qr)).T
                                                 @ target_r_full))
    print(f"[[ head decompose residual mean {hres.mean():.3f} rad (unrepresentable part)")
    print("[[ expressive chain orientation error (deg): what the eye actually sees")
    print(f"[[   {'chain':6s} {'within v4 limits':>22s} {'limits lifted':>18s} "
          f"{'cost of the LIMITS':>20s}")
    for _ci, _nm in enumerate(("head", "tail", "earL", "earR")):
        _a = np.degrees(ori_lim[:, _ci]); _b = np.degrees(ori_free[:, _ci])
        print(f"[[   {_nm:6s}  mean {_a.mean():5.1f} p95 {np.percentile(_a,95):5.1f}"
              f"   mean {_b.mean():5.1f} p95 {np.percentile(_b,95):5.1f}"
              f"      mean {(_a-_b).mean():5.1f} max {(_a-_b).max():5.1f}")
    EXPR = DOF_ORDER[12:21]
    lims_all = kin.all_limits()
    print("[[ expressive joint DEMAND vs v4 limit (what Ashley asks for, limits lifted):")
    for _c, _n in enumerate(EXPR):
        _j = 12 + _c
        _lo, _hi = lims_all[_j]
        _d = demand[:, _c]
        _over = float(np.mean((_d < _lo - 1e-4) | (_d > _hi + 1e-4)))
        _flag = "" if _over < 0.02 else (
            f"   <-- clipped on {100*_over:.0f}% of frames, "
            f"overshoot up to {np.degrees(max(_lo - _d.min(), _d.max() - _hi)):.0f} deg")
        print(f"     {_n:18s} demand [{np.degrees(_d.min()):+7.1f},{np.degrees(_d.max()):+7.1f}] "
              f"limit [{np.degrees(_lo):+7.1f},{np.degrees(_hi):+7.1f}] deg{_flag}")
    print(f"[[ visible ear residual mean L={ear_res[:,0].mean():.3f} R={ear_res[:,1].mean():.3f} rad")
    for fr in (1, 60, 94, 121, 127, 128, 180):
        where = np.where(frames == fr)[0]
        if not len(where):
            continue
        i = int(where[0])
        lq = mat_to_quat(el_t[i]); rq = mat_to_quat(er_t[i])
        print(f"[[ ear frame {fr:3d}: Ashley mapped delta L={np.round(lq,3)} "
              f"R={np.round(rq,3)} | v4 L={np.round(dof[i,17:19],3)} "
              f"R={np.round(dof[i,19:21],3)} | err deg "
              f"L={np.degrees(ear_res[i,0]):.1f} R={np.degrees(ear_res[i,1]):.1f}")

    # ---- continuity + rate bounds (10 rad/s legs / 8 expression) ---------------
    def max_vel(x, cols):
        return np.abs(np.gradient(x[:, cols], dt, axis=0)).max()

    # Refine the legs against their actual targets under per-frame velocity
    # bounds.  This preserves stance/swing poses far better than blurring every
    # joint with a Gaussian merely because a few frames are fast.
    step_leg = 10.0 * dt
    for direction in (range(1, T), range(T-2, -1, -1), range(1, T)):
        forward = direction.start < direction.stop
        for i in direction:
            nb = i - 1 if forward else i + 1
            for k, l in enumerate(LEGS):
                sl = slice(3*k, 3*k+3)
                lo = np.maximum(lims_leg[l][:, 0], dof[nb, sl] - step_leg)
                hi = np.minimum(lims_leg[l][:, 1], dof[nb, sl] + step_leg)
                q0 = np.minimum(np.maximum(dof[i, sl], lo + 1e-10), hi - 1e-10)
                q, _ = solve_leg_frame(i, k, q0, np.column_stack([lo, hi]),
                                       continuity=0.010)
                dof[i, sl] = q

    def project_rate(x, max_step, iterations=8):
        y = x.copy()
        for _ in range(iterations):
            for i in range(1, len(y)):
                y[i] = np.clip(y[i], y[i-1] - max_step, y[i-1] + max_step)
            for i in range(len(y)-2, -1, -1):
                y[i] = np.clip(y[i], y[i+1] - max_step, y[i+1] + max_step)
        return y

    dof[:, 12:17] = project_rate(dof[:, 12:17], a.expr_vel * dt)   # head + tail
    dof[:, 17:21] = project_rate(dof[:, 17:21], a.ear_vel * dt)    # ears

    # Recompute the final visible ear errors after expression rate projection.
    for i in range(T):
        H = kin.chain_rot(HEAD_CHAIN, dof[i, 12:15])
        target_l_full = el_t[i] @ (R0["h"] @ R0["l"])
        target_r_full = er_t[i] @ (R0["h"] @ R0["r"])
        ear_res[i, 0] = np.linalg.norm(rotvec_of(
            (H @ kin.chain_rot(LEAR_CHAIN, dof[i, 17:19])).T @ target_l_full))
        ear_res[i, 1] = np.linalg.norm(rotvec_of(
            (H @ kin.chain_rot(REAR_CHAIN, dof[i, 19:21])).T @ target_r_full))
    print(f"[[ continuity refinement: max velocity legs={max_vel(dof, list(range(12))):.2f} "
          f"expression={max_vel(dof, list(range(12,21))):.2f} rad/s")

    # ---- collision-support FK + small residual root correction ----------------
    ankles_w = np.zeros((T, 4, 3)); contacts_w = np.zeros((T, 4, 3))
    patch_w = np.zeros((T, 4, 3))
    lowest_w = np.zeros((T, 4, 3))
    knee_w = np.zeros((T, 4, 3)); sp_w = np.zeros((T, 4, 3))
    def recompute_points(root_pos):
        for ii in range(T):
            for kk, ll in enumerate(LEGS):
                SP, KP, AP, CP, _ = kin.leg_points(
                    ll, dof[ii, 3*kk:3*kk+3], support_hull=support_hull[ll],
                    world_R=R_b[ii], support_softness=0.001)
                LP = kin.leg_points(ll, dof[ii, 3*kk:3*kk+3],
                                    support_hull=support_hull[ll],
                                    world_R=R_b[ii])[3]
                sp_w[ii, kk] = root_pos[ii] + R_b[ii] @ SP
                knee_w[ii, kk] = root_pos[ii] + R_b[ii] @ KP
                ankles_w[ii, kk] = root_pos[ii] + R_b[ii] @ AP
                patch_w[ii, kk] = root_pos[ii] + R_b[ii] @ CP
                lowest_w[ii, kk] = root_pos[ii] + R_b[ii] @ LP
                contacts_w[ii, kk] = patch_w[ii, kk]
                contacts_w[ii, kk, 2] = lowest_w[ii, kk, 2]
                ankle_err[ii, kk] = np.linalg.norm(AP - ankle_tgt[ii, kk])
                knee_err[ii, kk] = np.linalg.norm(KP - knee_tgt[ii, kk])
                if ct[ii, kk]:
                    contact_err[ii, kk] = np.linalg.norm(
                        contacts_w[ii, kk] - contact_anchor[ii, kk])
                else:
                    contact_err[ii, kk] = np.linalg.norm(CP - contact_tgt[ii, kk])

    def stance_steps():
        s = []
        for i in range(1, T):
            p = ct[i] & ct[i-1]
            if p.any():
                s.extend(np.linalg.norm(contacts_w[i, p] - contacts_w[i-1, p], axis=1))
        return np.asarray(s)

    def cap_norm(v, cap):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v * np.minimum(1.0, cap / np.maximum(n, 1e-12))

    base_t = t_b.copy()
    root_off = np.zeros((T, 3)); root_raw = root_off.copy()
    recompute_points(t_b)
    slip_before_root = stance_steps()

    # Alternate a bounded global root residual with the physical leg solve. The
    # old code shifted the root once and never re-solved the legs, directly
    # invalidating its own anchors. Two alternations are sufficient here because
    # root translation enters contact position linearly.
    for _alt in range(2):
        recompute_points(base_t)  # q-dependent support with zero root residual
        rows = []; rhs = []
        w_contact, w_prior, w_vel, w_acc = 80.0, 8.0, 30.0, 15.0
        for i in range(T):
            for k in np.where(ct[i])[0]:
                row = np.zeros(T); row[i] = np.sqrt(w_contact)
                rows.append(row)
                rhs.append(np.sqrt(w_contact) * (contact_anchor[i, k] - contacts_w[i, k]))
            row = np.zeros(T); row[i] = np.sqrt(w_prior)
            rows.append(row); rhs.append(np.zeros(3))
        for i in range(1, T):
            row = np.zeros(T); row[i-1] = -np.sqrt(w_vel); row[i] = np.sqrt(w_vel)
            rows.append(row); rhs.append(np.zeros(3))
        for i in range(2, T):
            row = np.zeros(T); row[i-2] = np.sqrt(w_acc)
            row[i-1] = -2*np.sqrt(w_acc); row[i] = np.sqrt(w_acc)
            rows.append(row); rhs.append(np.zeros(3))
        root_raw = np.linalg.lstsq(np.vstack(rows), np.vstack(rhs), rcond=None)[0]
        root_off = cap_norm(root_raw, a.root_cap)
        max_step = a.root_step
        for _ in range(10):
            for i in range(1, T):
                dlt = root_off[i] - root_off[i-1]; n = np.linalg.norm(dlt)
                if n > max_step:
                    root_off[i] = root_off[i-1] + dlt * (max_step / n)
            for i in range(T-2, -1, -1):
                dlt = root_off[i] - root_off[i+1]; n = np.linalg.norm(dlt)
                if n > max_step:
                    root_off[i] = root_off[i+1] + dlt * (max_step / n)
            root_off = cap_norm(root_off, a.root_cap)
        t_b = base_t + root_off
        refresh_contact_targets(t_b)

        # Re-solve after moving the root so every stance target still refers to
        # the same absolute touchdown anchor.
        q_prev = {l: dof[0, 3*k:3*k+3].copy() for k, l in enumerate(LEGS)}
        for i in range(T):
            for k, l in enumerate(LEGS):
                sl = slice(3*k, 3*k+3)
                lo, hi = lims_leg[l][:, 0].copy(), lims_leg[l][:, 1].copy()
                if i > 0:
                    lo = np.maximum(lo, q_prev[l] - step_leg)
                    hi = np.minimum(hi, q_prev[l] + step_leg)
                q0 = np.minimum(np.maximum(dof[i, sl], lo + 1e-10), hi - 1e-10)
                q, _ = solve_leg_frame(i, k, q0, np.column_stack([lo, hi]),
                                       continuity=0.010)
                dof[i, sl] = q; q_prev[l] = q

    recompute_points(t_b)
    slip_after = stance_steps()
    support_speed = np.linalg.norm(np.gradient(contacts_w, dt, axis=0), axis=2)
    physical_height = lowest_w[:, :, 2] <= 0.005  # validated Stage-4 collision tolerance
    source_speed_cap = 0.06 * float(np.mean([kin.leg_reach(l) for l in LEGS]))
    # Scheduled stance must also be physically at the floor. Unexpected contacts
    # are reported only when the physical support is both low and source-stance slow.
    contacts_physical = physical_height & (ct | (support_speed <= source_speed_cap))
    schedule_match = 100.0 * float((contacts_physical == ct).mean())

    root_step = np.linalg.norm(np.diff(root_off, axis=0), axis=1)
    print(f"[[ contact anchor root solve: foot slip "
          f"{(slip_before_root.mean() if len(slip_before_root) else 0)*1000:.2f} -> "
          f"{(slip_after.mean() if len(slip_after) else 0)*1000:.2f} mm/stance-frame; "
          f"raw max {np.linalg.norm(root_raw,axis=1).max()*1000:.1f} mm -> bounded max "
          f"{np.linalg.norm(root_off,axis=1).max()*1000:.1f} mm; step max "
          f"{root_step.max()*1000:.1f} mm")
    print(f"[[ physical contact schedule: source/physical match {schedule_match:.1f}% | "
          f"source {int(ct.sum())} foot-frames, physical {int(contacts_physical.sum())}")
    print(f"[[ collision support z: min {lowest_w[:,:,2].min()*1000:+.1f} mm | "
          f"scheduled stance mean {lowest_w[ct][:,2].mean()*1000:+.2f} mm max "
          f"{lowest_w[ct][:,2].max()*1000:+.2f} mm")

    # ---- limits / saturation record --------------------------------------------
    lims = kin.all_limits()
    sat = np.zeros(21, int); sat_frames = {}
    for j in range(21):
        onlo = dof[:, j] <= lims[j, 0] + 1e-4
        onhi = dof[:, j] >= lims[j, 1] - 1e-4
        sat[j] = int((onlo | onhi).sum())
        if sat[j] > 0:
            fr = frames[np.where(onlo | onhi)[0]]
            sat_frames[DOF_ORDER[j]] = (int(sat[j]), int(fr[0]), int(fr[-1]))
    print(f"[[ joint-limit saturation (>0.5% of frames):")
    any_sat = False
    for j in range(21):
        pc = 100.0 * sat[j] / T
        if pc > 0.5:
            any_sat = True
            print(f"     {DOF_ORDER[j]:18s} {pc:5.1f}%  frames {sat_frames[DOF_ORDER[j]][1]}-"
                  f"{sat_frames[DOF_ORDER[j]][2]}  limit [{lims[j,0]:.2f},{lims[j,1]:.2f}]")
    if not any_sat:
        print("     none")
    print(f"[[ ankle IK error: mean {ankle_err.mean()*1000:.2f} mm  p95 "
          f"{np.percentile(ankle_err,95)*1000:.2f} mm  max {ankle_err.max()*1000:.2f} mm")
    print(f"[[ knee shape error: mean {knee_err.mean()*1000:.2f} mm | rigid contact error "
          f"mean {contact_err.mean()*1000:.2f} mm")

    # ---- root quaternions & velocities -----------------------------------------
    root_quat = np.array([mat_to_quat(R_b[i]) for i in range(T)])
    dof_vel = np.gradient(dof, dt, axis=0)
    print(f"[[ max |dof vel| legs {np.abs(dof_vel[:,:12]).max():.2f} (lim 10) | "
          f"expr {np.abs(dof_vel[:,12:]).max():.2f} (lim 8) rad/s")

    np.savez(a.out,
             fps=np.array(fps), frames=frames,
             dof_names=np.array(DOF_ORDER),
             root_pos=t_b.astype(np.float32),
             root_quat=root_quat.astype(np.float32),
             dof_positions=dof.astype(np.float32),
             dof_velocities=dof_vel.astype(np.float32),
             tips_world=lowest_w.astype(np.float32),
             ankles_world=ankles_w.astype(np.float32),
             contacts_world=lowest_w.astype(np.float32),
             planted_points_world=contacts_w.astype(np.float32),
             support_patch_world=patch_w.astype(np.float32),
             sp_world=sp_w.astype(np.float32),
             knees_world=knee_w.astype(np.float32),
             foot_err=ankle_err.astype(np.float32),
             ankle_err=ankle_err.astype(np.float32),
             knee_shape_err=knee_err.astype(np.float32),
             contact_err=contact_err.astype(np.float32),
             ankle_target_origin=ankle_tgt.astype(np.float32),
             knee_target_origin=knee_tgt.astype(np.float32),
             contact_target_origin=contact_tgt.astype(np.float32),
             contacts=contacts_physical,
             source_contacts=ct,
             physical_height_contacts=physical_height,
             support_speed=support_speed.astype(np.float32),
             contact_height_tolerance=np.array(0.005),
             contact_speed_tolerance=np.array(source_speed_cap),
             ground_z=np.array(0.0),
             contact_anchor_world=contact_anchor.astype(np.float32),
             contact_patch_softness=np.array(0.001),
             root_contact_offset=root_off.astype(np.float32),
             root_contact_offset_raw=root_raw.astype(np.float32),
             ear_orientation_error=ear_res.astype(np.float32),
             leg_map=np.array([f"{al}->{MAP[al]}" for al in ALEGS]),
             collision_hulls=np.array(a.collision_hulls),
             saturation=sat, expressive_demand=demand.astype(np.float32),
             expressive_ori_err=ori_lim.astype(np.float32),
             expressive_ori_err_unlimited=ori_free.astype(np.float32),
             source=d["source"])
    print(f"[[ wrote {a.out}")


main()
