"""Simulator-in-the-loop Stage-4 trajectory optimization for Bingo v4.

This is deliberately not Stage-5 learning.  It searches a small, interpretable
TMR control space (friction-constrained contact-wrench feed-forward scales) and
evaluates every candidate with the full Isaac rigid-body simulation.  The score
tracks the authored floating-base trajectory, orientation/yaw, contacts and joint
motion, and rejects falls or limit violations.

Temporal deformation is expected to be performed first with retime_segment.py;
`--warm-motion` is that temporally feasible result.  This separation mirrors the
paper's outer temporal deformation / inner model-based optimal-control structure.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation


def phase_sample(x, n):
    x = np.asarray(x)
    src = np.linspace(0.0, 1.0, len(x))
    dst = np.linspace(0.0, 1.0, n)
    flat = x.reshape(len(x), -1)
    out = np.stack([np.interp(dst, src, flat[:, j]) for j in range(flat.shape[1])], 1)
    return out.reshape((n,) + x.shape[1:])


def quat_error_deg(q, qr):
    # Both Blender/Isaac arrays are wxyz.  Rotation expects xyzw.
    q = np.asarray(q)[:, [1, 2, 3, 0]]
    qr = np.asarray(qr)[:, [1, 2, 3, 0]]
    return np.degrees((Rotation.from_quat(qr).inv() * Rotation.from_quat(q)).magnitude())


def yaw_unwrapped(q):
    q = np.asarray(q)[:, [1, 2, 3, 0]]
    return np.unwrap(Rotation.from_quat(q).as_euler("xyz")[:, 2])


def metrics(stage3, log):
    ref = np.load(stage3, allow_pickle=True)
    d = np.load(log, allow_pickle=True)
    n = len(d["root_pos"])
    rp_ref = phase_sample(ref["root_pos"], n)
    rq_ref = phase_sample(ref["root_quat"], n)
    rq_ref /= np.linalg.norm(rq_ref, axis=1, keepdims=True) + 1e-12
    rp = np.asarray(d["root_pos"], float)
    rq = np.asarray(d["root_quat"], float)
    pe = np.linalg.norm(rp - rp_ref, axis=1)
    oe = quat_error_deg(rq, rq_ref)
    y_ref, y = yaw_unwrapped(rq_ref), yaw_unwrapped(rq)
    yaw_err = np.degrees(np.abs(np.unwrap(y - y_ref)))

    actual = np.asarray(d["contacts"], bool)
    planned = np.asarray(d["planned_contacts"], bool)
    inter = np.logical_and(actual, planned).sum()
    union = np.logical_or(actual, planned).sum()
    iou = float(inter / union) if union else 1.0
    agree = float((actual == planned).mean())

    # Same authored-relative fall test as track_v4_physics.py.
    def up(qwxyz):
        rr = Rotation.from_quat(qwxyz[:, [1, 2, 3, 0]])
        return rr.apply(np.tile([0.0, 0.0, 1.0], (len(qwxyz), 1)))
    tilt_ref = np.degrees(np.arccos(np.clip((up(rq) * up(rq_ref)).sum(1), -1, 1)))
    fallen = (rp[:, 2] < 0.5 * np.maximum(rp_ref[:, 2], 1e-3)) | (tilt_ref > 45.0)
    fall_frame = int(np.where(fallen)[0][0]) if fallen.any() else -1

    vel = np.abs(d["q_vel"])
    torque = np.abs(d["torque"])
    vl = np.asarray(d["velocity_limits"] if "velocity_limits" in d.files else np.inf)
    el = np.asarray(d["effort_limits"] if "effort_limits" in d.files else np.inf)
    vel_violation = float(np.maximum(vel - vl[None], 0.0).max())
    effort_violation = float(np.maximum(torque - el[None], 0.0).max())
    qerr = np.asarray(d["q_err"], float)

    net_ref = rp_ref[-1] - rp_ref[0]
    net = rp[-1] - rp[0]
    hop_ref = float(rp_ref[:, 2].max() - rp_ref[0, 2])
    hop = float(rp[:, 2].max() - rp[0, 2])
    yaw_net_ref = float(np.degrees(y_ref[-1] - y_ref[0]))
    yaw_net = float(np.degrees(y[-1] - y[0]))

    # Dimensionless source-derived normalizers: no arbitrary pass thresholds.
    path_scale = max(0.05, float(np.linalg.norm(np.diff(rp_ref[:, :2], axis=0), axis=1).sum()))
    z_scale = max(0.01, float(np.ptp(rp_ref[:, 2])))
    ori_scale = max(10.0, float(np.degrees(np.abs(np.diff(y_ref)).sum())))
    score = (np.mean(np.abs(rp[:, :2] - rp_ref[:, :2])) / path_scale
             + np.mean(np.abs(rp[:, 2] - rp_ref[:, 2])) / z_scale
             + np.mean(oe) / ori_scale
             + (1.0 - iou)
             + 0.5 * float(qerr.mean()))
    if fall_frame >= 0:
        score += 20.0 + 20.0 * (1.0 - fall_frame / max(1, n - 1))
    score += 10.0 * vel_violation + 10.0 * effort_violation
    return dict(score=float(score), frames=n, fall_frame=fall_frame,
                root_error_mean_m=float(pe.mean()), root_error_max_m=float(pe.max()),
                orientation_error_mean_deg=float(oe.mean()),
                orientation_error_max_deg=float(oe.max()),
                yaw_error_mean_deg=float(yaw_err.mean()),
                contact_iou=iou, contact_agreement=agree,
                joint_error_mean_rad=float(qerr.mean()),
                joint_error_max_rad=float(qerr.max()),
                velocity_violation_rad_s=vel_violation,
                effort_violation_nm=effort_violation,
                net_displacement_ref_m=net_ref.tolist(),
                net_displacement_actual_m=net.tolist(),
                hop_ref_m=hop_ref, hop_actual_m=hop,
                yaw_net_ref_deg=yaw_net_ref, yaw_net_actual_deg=yaw_net)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage3", required=True)
    ap.add_argument("--warm-motion", required=True,
                    help="locally retimed, kinematically faithful Stage-4 warm start")
    ap.add_argument("--contacts", required=True)
    ap.add_argument("--out", required=True, help="best dynamics-aware reference npz")
    ap.add_argument("--result", required=True, help="best full-physics rollout npz")
    ap.add_argument("--report", required=True, help="machine-readable search report json")
    ap.add_argument("--isaaclab", default="/home/hassaan/robotics/IsaacLab")
    ap.add_argument("--balance-args", default="--base-balance -0.6 --balance-windows 115-140,168-193")
    ap.add_argument("--max-candidates", type=int, default=13)
    a = ap.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tracker = os.path.join(repo, "rl", "tools", "track_v4_physics.py")
    builder = os.path.join(repo, "stage4", "dynamic_retarget.py")
    launcher = os.path.join(a.isaaclab, "isaaclab.sh")
    # Zero is the trusted warm start. Remaining points are a bounded coordinate
    # search followed by combined candidates; this is reproducible and restartable.
    candidates = [
        (0.0, 1.0, 1.0, 1.0),
        (-0.15, 1.0, 1.0, 1.0), (0.15, 1.0, 1.0, 1.0),
        (-0.30, 1.0, 1.0, 1.0), (0.30, 1.0, 1.0, 1.0),
        (-0.20, 1.5, 1.0, 1.0), (0.20, 1.5, 1.0, 1.0),
        (-0.20, 1.0, 1.5, 1.0), (0.20, 1.0, 1.5, 1.0),
        (-0.20, 1.0, 1.0, 1.5), (0.20, 1.0, 1.0, 1.5),
        (-0.20, 1.5, 1.5, 1.5), (0.20, 1.5, 1.5, 1.5),
    ][:a.max_candidates]
    balance = a.balance_args.split() if a.balance_args else []
    records = []
    best = None
    with tempfile.TemporaryDirectory(prefix="bingo_tmr_") as td:
        for k, (force, horiz, vert, yaw) in enumerate(candidates):
            mot = os.path.join(td, f"candidate_{k:02d}.npz")
            sim_out = os.path.join(td, f"sim_{k:02d}")
            os.makedirs(sim_out)
            make = ["python3", builder, "--motion", a.warm_motion,
                    "--contacts", a.contacts, "--out", mot,
                    "--force-scale", str(force), "--horizontal-scale", str(horiz),
                    "--vertical-scale", str(vert), "--yaw-scale", str(yaw)]
            subprocess.run(make, cwd=repo, check=True, stdout=subprocess.DEVNULL)
            run = [launcher, "-p", tracker, "--motion", mot, "--out", sim_out,
                   "--headless", "--contact-source", a.contacts] + balance
            env = dict(os.environ); env["TERM"] = "xterm"
            proc = subprocess.run(run, cwd=a.isaaclab, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True)
            log = os.path.join(sim_out, f"candidate_{k:02d}_stage4.npz")
            rec = dict(candidate=k, force_scale=force, horizontal_scale=horiz,
                       vertical_scale=vert, yaw_scale=yaw, returncode=proc.returncode)
            if os.path.exists(log):
                rec.update(metrics(a.stage3, log))
                if best is None or rec["score"] < best[0]:
                    best = (rec["score"], mot, log, rec.copy())
            else:
                rec.update(score=float("inf"), error="Isaac produced no rollout",
                           tail="\n".join(proc.stdout.splitlines()[-12:]))
            records.append(rec)
            print(f"[[ candidate {k:02d} force={force:+.2f} h/v/y={horiz:g}/{vert:g}/{yaw:g} "
                  f"score={rec['score']:.4f} fall={rec.get('fall_frame','ERROR')}", flush=True)
        if best is None:
            raise SystemExit("no Isaac candidate completed; see report/GPU availability")
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.result)), exist_ok=True)
        shutil.copy2(best[1], a.out)
        shutil.copy2(best[2], a.result)

    report = dict(method="TMR temporal warm start + friction-constrained contact-wrench "
                         "allocation + full-Isaac coordinate search",
                  stage3=os.path.abspath(a.stage3),
                  warm_motion=os.path.abspath(a.warm_motion),
                  best=best[3], candidates=records)
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    with open(a.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[[ best candidate {best[3]['candidate']} score {best[0]:.4f}")
    print(f"[[ wrote {a.out}, {a.result}, {a.report}")


if __name__ == "__main__":
    main()
