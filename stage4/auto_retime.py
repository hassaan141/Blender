"""Locally retime exactly the windows that are contact-wrench INFEASIBLE, and only
those - the STMR idea, aimed by a physical test instead of by hand.

stage4/wrench_refine.py --lp says, per frame, whether ANY set of ground reaction
forces inside the friction cone can produce the motion's own momentum rate. Where
it says no, the fix that does not touch a single pose is to revisit those frames on
a slower clock: accelerations scale by 1/f^2 and the required centre of pressure
walks back inside the support. Everything outside the window keeps Ashley's tempo.

Windows are padded and merged first, so a fix is not applied to three frames in the
middle of one gesture, and the pass repeats until either every window is feasible or
--rounds is spent. The report says exactly which windows were slowed and by how
much, because that is the cost paid against the authored performance.

  python3 stage4/auto_retime.py --motion motions/deadpan_v4.npz \
      --out motions/deadpan_v4_retimed.npz --factor 2.0 --rounds 4
"""
import argparse, itertools, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wrench_refine import centroidal, evaluate


def windows(bad, T, pad, gap):
    if not len(bad):
        return []
    runs = []
    for _, g in itertools.groupby(enumerate(bad), lambda t: t[1] - t[0]):
        gg = [x[1] for x in g]
        runs.append([max(0, gg[0] - pad), min(T - 1, gg[-1] + pad)])
    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def infeasible(path, mu, tol):
    D = centroidal(path, tol)
    hdot = np.gradient(D["h"], D["dt"], axis=0)
    ev = evaluate(D["C"], hdot, D["M"], D["sup"], D["dt"], mu)
    ok = (ev["margin"] > 0) & (ev["mu"] <= mu) & ~ev["airborne"]
    return np.where(~ok)[0], D["T"], float(ok.mean())


def apply_retime(path, out, wins, factor):
    m = np.load(path, allow_pickle=True); d = {k: m[k] for k in m.files}
    T = m["dof_positions"].shape[0]
    src = []
    prev = 0
    for s, e in wins:
        src.append(np.arange(prev, s, dtype=float))
        n = max(2, int(round((e - s) * factor)))
        src.append(np.linspace(s, e, n, endpoint=False))
        prev = e
    src.append(np.arange(prev, T, dtype=float))
    src = np.concatenate(src)
    T2 = len(src)

    # CUBIC, not linear. np.interp gives a piecewise-linear curve whose second
    # difference is impulsive at every original sample, and the whole point of this
    # pass is to look at accelerations: measured on DeadPan, linear resampling made
    # the wrench-feasibility score FALL from 93% to 66% while nominally slowing the
    # motion down. A C2 spline preserves the accelerations the retime is supposed to
    # scale.
    from scipy.interpolate import CubicSpline

    def resamp(x):
        x = np.asarray(x, float); flat = x.reshape(T, -1)
        cs = CubicSpline(np.arange(T), flat, axis=0)
        return np.asarray(cs(src)).reshape((T2,) + x.shape[1:])

    smooth = ("dof_positions", "body_positions", "body_rotations", "root_pos",
              "root_quat", "head_tail_positions", "ear_positions")
    idx = np.clip(np.round(src).astype(int), 0, T - 1)
    for k in list(d):
        v = np.asarray(d[k])
        if v.ndim >= 1 and v.shape[0] == T:
            if k in smooth:
                d[k] = resamp(v).astype(np.float32)
            elif k == "dof_velocities":
                continue
            else:
                d[k] = v[idx] if (v.dtype == bool or v.dtype.kind in "iub") \
                    else resamp(v).astype(v.dtype)
    for k in ("body_rotations", "root_quat"):
        if k in d:
            qq = np.asarray(d[k], float)
            qq /= (np.linalg.norm(qq, axis=-1, keepdims=True) + 1e-12)
            d[k] = qq.astype(np.float32)
    dt = 1.0 / float(m["fps"])
    d["dof_velocities"] = np.gradient(np.asarray(d["dof_positions"], float),
                                      dt, axis=0).astype(np.float32)
    np.savez(out, **d)
    return T2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--gap", type=int, default=8, help="merge windows closer than this")
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--contact-tol", type=float, default=0.006)
    a = ap.parse_args()

    cur = a.motion
    tmp = a.out + ".tmp.npz"
    for r in range(a.rounds):
        bad, T, frac = infeasible(cur, a.mu, a.contact_tol)
        wins = windows(bad, T, a.pad, a.gap)
        print(f"[[ round {r}: {T} frames, wrench-feasible {100*frac:.0f}% | "
              f"{len(wins)} window(s) to slow: "
              + (", ".join(f"{s}-{e}" for s, e in wins[:10]) or "none")
              + (" ..." if len(wins) > 10 else ""))
        if not wins:
            break
        T2 = apply_retime(cur, tmp, wins, a.factor)
        print(f"[[   x{a.factor} on those windows: {T} -> {T2} frames "
              f"({T/24:.2f}s -> {T2/24:.2f}s at 24 fps)")
        cur = tmp
    bad, T, frac = infeasible(cur, a.mu, a.contact_tol)
    m = np.load(cur, allow_pickle=True)
    np.savez(a.out, **{k: m[k] for k in m.files})
    if os.path.exists(tmp):
        os.remove(tmp)
    src = np.load(a.motion, allow_pickle=True)["dof_positions"].shape[0]
    print(f"[[ FINAL: {T} frames (was {src}, +{100*(T/src-1):.0f}% duration) | "
          f"wrench-feasible {100*frac:.0f}% | remaining infeasible frames {len(bad)}")
    print(f"[[ wrote {a.out}")


if __name__ == "__main__":
    main()
