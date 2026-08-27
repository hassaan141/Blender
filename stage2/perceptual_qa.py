"""Stage 2 QA on the quantities an EYE reads, Ashley vs the v4 retarget.

Numeric gates (foot slip, ground penetration, IK residual) prove a clip is not
broken. They cannot say it does not LOOK like Ashley's performance. This does, by
measuring both characters in their own anatomical body frame, on
proportion-robust quantities, and reporting the contiguous frame windows where
they diverge - so a finding turns straight into a fix with a frame range.

Method adopted from research/mixamo-llm-mocap (pipeline/compare_reference.py):
compare on eye-readable quantities, normalise proportion away, group findings
into windows. The quantity list is Bingo's own.

Per frame, in the character's own anatomical frame
(forward = front-hip-centre minus back-hip-centre, up = forward x (left - right)):

  crouch      body height above its own ground plane, / leg length
  pitch, roll body attitude in degrees (heading is compared as a yaw RATE, since
              the two characters do not share a world heading)
  paw f/l/v   each paw in body coordinates, / leg length - the stance shape
  gaze        head direction in body coordinates, yaw + elevation, degrees
  tail        tail direction, yaw + elevation
  ear L/R     each ear direction, yaw + elevation

Every length is divided by that character's OWN leg length, so the 6% scale
difference between the rigs cannot masquerade as a pose error. The head/tail/ear
chains use different rest axes on the two rigs, so those are compared as the
change from frame 0 - the animated part, which is what was authored.

  python3 stage2/perceptual_qa.py --source stage2/out/cheeky_source.npz \
      --motion motions/cheeky_v4.npz [--tol-scale 1.0] [--detail 40-60]
"""
import argparse, itertools, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "stage4"))
from v4_kinematics import V4Kin, LEGS, axis_rot, quat_to_mat
from contact_model import ContactModel

URDF = os.path.join(HERE, "..", "URDF", "bingo_urdf v4_w_ear_joints", "urdf",
                    "bingo_urdf_w_ear_joints_physics.urdf")
ALEGS = ["aFL", "aFR", "aBL", "aBR"]
MAP = {"fl": "aFL", "fr": "aFR", "bl": "aBL", "br": "aBR"}
TOL = {"crouch": 0.06, "pitch": 8.0, "roll": 8.0, "yaw_rate": 3.0,
       "f": 0.08, "l": 0.08, "v": 0.06,
       "gaze_yaw": 12.0, "gaze_el": 12.0, "tail_yaw": 15.0, "tail_el": 15.0,
       "earL_yaw": 20.0, "earL_el": 20.0, "earR_yaw": 20.0, "earR_el": 20.0}


def anat(fc, bc, lc, rc):
    f = fc - bc; f /= np.linalg.norm(f) + 1e-12
    u = np.cross(f, lc - rc); u /= np.linalg.norm(u) + 1e-12
    return np.column_stack([f, np.cross(u, f), u])


def wrap(a):
    """Fold an angle difference into [-180, 180] - a yaw that passes through the
    +/-180 seam is otherwise reported as a 360 deg error."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def dir_angles(v):
    n = np.linalg.norm(v) + 1e-12
    return np.array([np.degrees(np.arctan2(v[1], v[0])),
                     np.degrees(np.arcsin(np.clip(v[2] / n, -1, 1)))])


def ashley(path):
    s = np.load(path, allow_pickle=True)
    # Ashley's rig is LEFT-handed relative to the robot (her forward is +Y, left
    # +X). solve_spatial_retarget reflects the source across the YZ plane exactly
    # once at load; do the same here or her anatomical "up" comes out inverted and
    # every roll reads ~180 deg off. p -> F p, R -> F R F (stays a proper rotation).
    F = np.diag([-1.0, 1.0, 1.0])
    rf = lambda P: P @ F
    rfR = lambda R: F @ R @ F
    sy = {l: rf(s[f"sy_{l}"].astype(float)) for l in ALEGS}
    toe = {l: rf(s[f"toe_{l}"].astype(float)) for l in ALEGS}
    body = rf(s["body_pos"].astype(float))
    chains = {"gaze": s["head_quat"], "tail": s["tail_quat"],
              "earL": s["earl_quat"], "earR": s["earr_quat"]}
    R = {k: np.array([rfR(quat_to_mat(q)) for q in v.astype(float)]) for k, v in chains.items()}
    T = len(body)
    L = float(s["rest_lengths"].astype(float).sum(1).mean())
    g = np.percentile(np.concatenate([toe[l][:, 2] for l in ALEGS]), 2.0)
    sc = {k: np.zeros(T) for k in ("crouch", "pitch", "roll", "yaw")}
    paw = {l: np.zeros((T, 3)) for l in LEGS}
    ang = {k: np.zeros((T, 2)) for k in R}
    for i in range(T):
        A = anat(0.5*(sy["aFL"][i]+sy["aFR"][i]), 0.5*(sy["aBL"][i]+sy["aBR"][i]),
                 0.5*(sy["aFL"][i]+sy["aBL"][i]), 0.5*(sy["aFR"][i]+sy["aBR"][i]))
        sc["crouch"][i] = (body[i, 2] - g) / L
        sc["pitch"][i] = np.degrees(np.arcsin(np.clip(-A[2, 0], -1, 1)))
        sc["roll"][i] = np.degrees(np.arctan2(A[2, 1], A[2, 2]))
        sc["yaw"][i] = np.degrees(np.arctan2(A[1, 0], A[0, 0]))
        # relative to that leg's OWN hip pivot, not the hip centroid: the v4 hips
        # sit ~22% wider than Ashley's, and measuring from the centroid turns that
        # fixed layout difference into a permanent fake "stance too wide" finding.
        for l in LEGS:
            paw[l][i] = A.T @ (toe[MAP[l]][i] - sy[MAP[l]][i]) / L
        for k in R:
            ang[k][i] = dir_angles(A.T @ (R[k][i] @ np.array([0., 1., 0.])))
    for k in ang:
        ang[k] = wrap(ang[k] - ang[k][0])
    return dict(sc=sc, paw=paw, ang=ang, T=T, L=L)


def robot(path):
    kin = V4Kin(URDF); cm = ContactModel()
    ch = {}
    for n, j in kin.j.items():
        ch.setdefault(j["parent"], []).append(n)
    m = np.load(path, allow_pickle=True)
    q = m["dof_positions"].astype(float); nm = [str(x) for x in m["dof_names"]]
    rp = (m["root_pos"] if "root_pos" in m.files else m["body_positions"][:, 0]).astype(float)
    rq = (m["root_quat"] if "root_quat" in m.files else m["body_rotations"][:, 0]).astype(float)
    T = len(q)
    L = float(np.mean([kin.leg_reach(l) for l in LEGS]))
    sc = {k: np.zeros(T) for k in ("crouch", "pitch", "roll", "yaw")}
    paw = {l: np.zeros((T, 3)) for l in LEGS}
    ang = {k: np.zeros((T, 2)) for k in ("gaze", "tail", "earL", "earR")}
    LINK = {"gaze": "head_roll", "tail": "tail_yaw",
            "earL": "l_ear_roll", "earR": "r_ear_roll"}
    for i in range(T):
        d = {n: q[i, nm.index(n)] for n in nm}
        fr = {"origin": (quat_to_mat(rq[i]), rp[i].copy())}; st = ["origin"]
        while st:
            par = st.pop(); Rp, pp = fr[par]
            for jn in ch.get(par, []):
                J = kin.j[jn]; pj = pp + Rp @ J["xyz"]
                fr[J["child"]] = (Rp @ J["R"] @ axis_rot(J["axis"], d.get(jn, 0.0)), pj)
                st.append(J["child"])
        A = fr["origin"][0]                 # the URDF origin IS the anatomical frame
        sc["crouch"][i] = rp[i, 2] / L
        sc["pitch"][i] = np.degrees(np.arcsin(np.clip(-A[2, 0], -1, 1)))
        sc["roll"][i] = np.degrees(np.arctan2(A[2, 1], A[2, 2]))
        sc["yaw"][i] = np.degrees(np.arctan2(A[1, 0], A[0, 0]))
        for l in LEGS:
            Rk, pk = fr[f"{l}_knee"]; w = cm.hull[f"{l}_knee"] @ Rk.T + pk
            hip = rp[i] + A @ kin.sy_pos(l)
            paw[l][i] = A.T @ (w[int(np.argmin(w[:, 2]))] - hip) / L
        for k, ln in LINK.items():
            ang[k][i] = dir_angles(A.T @ (fr[ln][0] @ np.array([1., 0., 0.])))
    for k in ang:
        ang[k] = wrap(ang[k] - ang[k][0])
    return dict(sc=sc, paw=paw, ang=ang, T=T, L=L)


def windows(bad, minlen=3):
    out = []
    for _, g in itertools.groupby(enumerate(bad), lambda t: t[1] - t[0]):
        gg = [x[1] for x in g]
        if len(gg) >= minlen:
            out.append((gg[0], gg[-1]))
    return out


def resample(D, T2):
    """Stretch a measurement set onto T2 frames (for retimed clips)."""
    T = D["T"]
    if T == T2:
        return D
    src = np.linspace(0, T - 1, T2)
    def rs(x):
        x = np.asarray(x, float); f = x.reshape(T, -1)
        o = np.stack([np.interp(src, np.arange(T), f[:, c]) for c in range(f.shape[1])], 1)
        return o.reshape((T2,) + x.shape[1:])
    return dict(sc={k: rs(v) for k, v in D["sc"].items()},
                paw={k: rs(v) for k, v in D["paw"].items()},
                ang={k: rs(v) for k, v in D["ang"].items()}, T=T2, L=D["L"])


def compare(src, mot, tol_scale=1.0, detail=None, quiet=False):
    A = ashley(src); R = robot(mot)
    # A retimed clip has more frames than the source it came from; compare like for
    # like by stretching the SOURCE onto the robot's timeline. Tempo is reported as
    # a separate, deliberate modification in the clip's recipe - it must not show up
    # here as a pose error.
    if A["T"] != R["T"]:
        A = resample(A, R["T"])
    T = min(A["T"], R["T"])
    D = {}
    D["crouch"] = R["sc"]["crouch"][:T] - A["sc"]["crouch"][:T]
    for k in ("pitch", "roll"):
        D[k] = R["sc"][k][:T] - A["sc"][k][:T]
    ya = np.degrees(np.unwrap(np.radians(A["sc"]["yaw"][:T])))
    yr = np.degrees(np.unwrap(np.radians(R["sc"]["yaw"][:T])))
    D["yaw_rate"] = np.gradient(yr) - np.gradient(ya)
    for l in LEGS:
        for j, s in enumerate(("f", "l", "v")):
            D[f"{l}_{s}"] = R["paw"][l][:T, j] - A["paw"][l][:T, j]
    for k in ("gaze", "tail", "earL", "earR"):
        D[f"{k}_yaw"] = wrap(R["ang"][k][:T, 0] - A["ang"][k][:T, 0])
        D[f"{k}_el"] = wrap(R["ang"][k][:T, 1] - A["ang"][k][:T, 1])
    if quiet:
        return D, T
    print(f"[[ {os.path.basename(mot)} vs {os.path.basename(src)}   {T} frames | "
          f"leg length: Ashley {A['L']:.3f} src-units, v4 {R['L']*1000:.1f} mm")
    if A["T"] != R["T"]:
        print(f"[[ note: {A['T']} source frames vs {R['T']} robot frames")
    # BIAS is the clip-constant part of the difference: the two rigs' landmarks are
    # not the same physical point (Ashley's toe is her paw tip, the robot's is the
    # lowest vertex of the shank hull) and their hips sit differently, so a fixed
    # offset is morphology, not a tracking error. VARY is the part that changes over
    # the clip - that is the authored motion, and it is what the windows flag.
    print(f"[[ {'quantity':12s} {'bias':>8s} {'vary':>8s} {'p95':>8s} {'max':>8s} "
          f"{'tol':>6s}  windows over tol (on the de-biased signal)")
    for k, v in D.items():
        base = k.split("_")[-1] if k.split("_")[0] in LEGS else k
        tol = TOL.get(base, 10.0) * tol_scale
        bias = float(np.median(v)); dv = v - bias
        av = np.abs(dv)
        rm = np.convolve(av, np.ones(5) / 5, mode="same")
        w = windows(np.where(rm > tol)[0])
        flag = "" if not w else "  " + ", ".join(f"{x}-{y}" for x, y in w[:8]) + \
               (" ..." if len(w) > 8 else "")
        print(f"[[ {k:12s} {bias:+8.3f} {av.mean():8.3f} {np.percentile(av,95):8.3f} "
              f"{av.max():8.3f} {tol:6.2f}{flag}")
    if detail:
        f0, f1 = (int(x) for x in detail.split("-", 1))
        keys = list(D)
        print("  frame " + " ".join(f"{k[:9]:>9s}" for k in keys))
        for i in range(max(0, f0), min(T - 1, f1) + 1):
            print(f"  {i:5d} " + " ".join(f"{D[k][i]:9.3f}" for k in keys))
    return D, T


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--tol-scale", type=float, default=1.0)
    ap.add_argument("--detail", default=None)
    a = ap.parse_args()
    compare(a.source, a.motion, a.tol_scale, a.detail)
