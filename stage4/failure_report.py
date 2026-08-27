"""Where and HOW a Stage-4 physics run left the reference.

Reads the log that rl/tools/track_v4_physics.py writes and answers the three
questions that decide which stage owns the problem:

  WHEN   the first frame where the body pose leaves the reference for good
  WHAT   which of roll / pitch / yaw / height diverges first, and in which sign
  WHY    what the feet were doing at that moment - which paw left the floor,
         which one never loaded, how much a planted paw skated, and whether any
         joint was saturated or lagging

Roll and pitch are compared IN THE REFERENCE'S OWN FRAME, so a clip that is
authored tilted (Eccentric sits at 47-78 deg) is not scored as falling.

  python3 stage4/failure_report.py stage4/out/laidback_v4_stage4.npz [--window 20]
"""
import argparse, os
import numpy as np


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--window", type=int, default=16, help="frames of detail to print")
    ap.add_argument("--tilt", type=float, default=15.0,
                    help="degrees of body-attitude error that counts as leaving")
    a = ap.parse_args()
    d = np.load(a.log, allow_pickle=True)
    names = [str(x) for x in d["joint_names"]]
    T = len(d["root_pos"])
    Ra = np.array([quat_to_R(q) for q in d["root_quat"]])
    Rr = np.array([quat_to_R(q) for q in d["root_quat_ref"]])
    # body-attitude error expressed in the REFERENCE body frame: roll about its
    # forward axis, pitch about its left axis, yaw about its up axis
    E = np.einsum("tji,tjk->tik", Rr, Ra)                 # Rr^T Ra
    err_roll = np.degrees(np.arctan2(E[:, 2, 1], E[:, 2, 2]))
    err_pitch = np.degrees(np.arcsin(np.clip(-E[:, 2, 0], -1, 1)))
    err_yaw = np.degrees(np.arctan2(E[:, 1, 0], E[:, 0, 0]))
    dz = (d["root_pos"][:, 2] - d["root_pos_ref"][:, 2]) * 1000.0
    tot = np.degrees(np.arccos(np.clip((np.trace(E, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    bad = np.where(tot > a.tilt)[0]
    first = int(bad[0]) if len(bad) else -1
    print(f"[[ {os.path.basename(a.log)}  {T} frames")
    print(f"[[ body attitude error (in the reference's own frame): "
          f"roll {np.abs(err_roll).mean():.1f} pitch {np.abs(err_pitch).mean():.1f} "
          f"yaw {np.abs(err_yaw).mean():.1f} deg mean")
    if first < 0:
        print(f"[[ never exceeds {a.tilt:g} deg of attitude error - no departure")
    else:
        dom = max((("roll", abs(err_roll[first])), ("pitch", abs(err_pitch[first])),
                   ("yaw", abs(err_yaw[first]))), key=lambda t: t[1])
        print(f"[[ first departure (>{a.tilt:g} deg): frame {first}, dominated by "
              f"{dom[0].upper()} ({err_roll[first]:+.1f} roll / {err_pitch[first]:+.1f} "
              f"pitch / {err_yaw[first]:+.1f} yaw)")
    s = max(0, (first if first >= 0 else T) - a.window)
    e = min(T, s + 2 * a.window)
    ct = d["contacts"]; pz = d["paw_z"] * 1000.0
    ref_ct = d["q_ref"]  # placeholder to keep the tuple shape stable
    print("  frame  droll dpitch  dyaw    dz | paw z (fl fr bl br) mm | n | qerr max  tau max")
    for i in range(s, e):
        print(f"  {i:5d} {err_roll[i]:+6.1f} {err_pitch[i]:+6.1f} {err_yaw[i]:+6.1f} "
              f"{dz[i]:+6.1f} | " + " ".join(f"{v:+7.1f}" for v in pz[i]) +
              f" | {int(ct[i].sum())} | {d['q_err'][i].max():8.3f} "
              f"{np.abs(d['torque'][i]).max():8.2f}")
    j = int(np.argmax(d["q_err"][s:e].max(0)))
    print(f"[[ worst joint in that window: {names[j]} "
          f"max {d['q_err'][s:e, j].max():.3f} rad")
    sl = []
    for i in range(1, T):
        p = ct[i] & ct[i - 1]
        if p.any():
            sl.append(np.linalg.norm(d["paw_xy"][i][p] - d["paw_xy"][i-1][p], axis=1).max())
    if sl:
        sl = np.array(sl)
        print(f"[[ planted-paw skate over the clip: mean {sl.mean()*1000:.2f} mm/frame, "
              f"worst frame {int(np.argmax(sl))+1} ({sl.max()*1000:.1f} mm)")


if __name__ == "__main__":
    main()
