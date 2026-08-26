"""Stage 2.4 - detect foot contacts from the source paw motion.

A paw is PLANTED when it is both low (near the source ground) and slow
(small world speed). Uses two-threshold hysteresis plus minimum segment
lengths so a single noisy frame cannot flip the state.

Run (system python):
  python3 stage2/detect_contacts.py --keypoints stage2/out/cheeky_source_keypoints.npz \
      --out stage2/out/cheeky_contacts.npz
"""
import argparse, numpy as np

ALEGS = ["aFL", "aFR", "aBL", "aBR"]


def hysteresis(low_ok, strong, min_plant, min_swing):
    """Combine a permissive mask (low_ok) and a strict mask (strong) into a
    stable boolean contact track, then enforce minimum run lengths."""
    T = len(low_ok)
    st = np.zeros(T, bool)
    cur = strong[0]
    for i in range(T):
        if cur:                      # stay planted while merely low
            cur = low_ok[i]
        else:                        # only enter contact when strictly planted
            cur = strong[i]
        st[i] = cur
    # remove runs shorter than the minimum (scan every run; flip only the short
    # ones whose value == want, and never the leading/trailing run)
    for want, mn in ((True, min_plant), (False, min_swing)):
        i = 0
        while i < T:
            j = i
            while j < T and st[j] == st[i]:
                j += 1
            if st[i] == want and (j - i) < mn and i > 0 and j < T:
                st[i:j] = not want
            i = j
    return st


def intervals(mask, frames):
    out = []
    i = 0
    T = len(mask)
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            out.append((int(frames[i]), int(frames[j-1])))
            i = j
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoints", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height-frac", type=float, default=0.18,
                    help="contact height band as a fraction of mean leg length")
    ap.add_argument("--speed-frac", type=float, default=0.06,
                    help="contact speed cap as a fraction of leg-length per second")
    a = ap.parse_args()

    d = np.load(a.keypoints, allow_pickle=True)
    fps = float(d["fps"]); frames = d["frames"]; dt = 1.0 / fps
    T = len(frames)
    rest = d["rest_lengths"]                      # (4,2) thigh,shank per leg
    leglen = float(rest.sum(1).mean())            # mean total leg length (units)

    # Contact belongs to Ashley's toe/paw-bone tail.  The independently moving
    # ankle HEAD is the IK endpoint and must not replace the contact signal.
    paw = {l: d[f"toe_{l}"] for l in ALEGS}
    ground = np.percentile(np.concatenate([paw[l][:, 2] for l in ALEGS]), 3.0)

    h_band = a.height_frac * leglen
    v_cap = a.speed_frac * leglen                 # units / s

    contacts = np.zeros((T, 4), bool)
    diag = {}
    for k, l in enumerate(ALEGS):
        z = paw[l][:, 2] - ground
        vel = np.linalg.norm(np.gradient(paw[l], dt, axis=0), axis=1)   # units/s
        low_ok = (z < 1.6 * h_band) & (vel < 2.0 * v_cap)     # permissive (hold)
        strong = (z < h_band) & (vel < v_cap)                 # strict (enter)
        st = hysteresis(low_ok, strong, min_plant=max(2, int(0.06*fps)),
                        min_swing=max(2, int(0.05*fps)))
        contacts[:, k] = st
        diag[l] = (z, vel)

    print(f"[[ leglen {leglen:.2f} u  height band <{h_band:.2f} u  speed cap <{v_cap:.2f} u/s"
          f"  ground z={ground:.2f}")
    for k, l in enumerate(ALEGS):
        duty = contacts[:, k].mean()
        iv = intervals(contacts[:, k], frames)
        print(f"[[ {l}: duty {duty*100:4.0f}%  {len(iv)} plant(s): {iv}")

    np.savez(a.out, contacts=contacts, aleg_order=np.array(ALEGS),
             frames=frames, fps=np.array(fps), ground=np.array(ground))
    print(f"[[ wrote {a.out}")


main()
