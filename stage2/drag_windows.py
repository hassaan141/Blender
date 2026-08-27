"""Find the frames where Ashley's paw is ON THE FLOOR BUT MOVING - a DRAG - and emit
them as a --force-planted specification for stage4/ground_fix.py.

stage2/detect_contacts.py marks a paw planted only when it is both LOW and SLOW. The
speed gate is right for its own job (a world anchor must not be pinned to a foot that
is travelling), but it means a dragging paw is filed as swing, and nothing downstream
then keeps it on the floor. Eccentric is the case that exposed it: Bingo shuffles
forward by dragging both front paws, Ashley's front toes sit within 5 mm of her floor
on 55-57% of frames while travelling at ~190 mm/s, and the schedule reports 25%. The
retarget duly left them in the air and the body slid forward with nothing touching.

The distinction this restores:
    planted  = low AND slow  -> anchor to a fixed world point (contacts, unchanged)
    dragging = low AND fast  -> hold on the floor, but let it slide  (this file)
    swing    = high          -> free

  python3 stage2/drag_windows.py --keypoints stage2/out/eccentric_source.npz \
      --contacts stage2/out/eccentric_contacts.npz --band 0.05
"""
import argparse, itertools
import numpy as np

ALEGS = ["aFL", "aFR", "aBL", "aBR"]
MAP = {"aFL": "fl", "aFR": "fr", "aBL": "bl", "aBR": "br"}

ap = argparse.ArgumentParser()
ap.add_argument("--keypoints", required=True)
ap.add_argument("--contacts", required=True)
ap.add_argument("--band", type=float, default=0.05,
                help="height band as a fraction of leg length. A paw whose toe is "
                     "below this above the clip's floor counts as touching, however "
                     "fast it is moving.")
ap.add_argument("--min-len", type=int, default=3, help="ignore shorter windows")
ap.add_argument("--gap", type=int, default=3, help="merge windows closer than this")
ap.add_argument("--legs", default=None, help="restrict to e.g. fl,fr")
a = ap.parse_args()

s = np.load(a.keypoints, allow_pickle=True)
c = np.load(a.contacts, allow_pickle=True)
g = float(c["ground"]); ct = c["contacts"]; order = [str(x) for x in c["aleg_order"]]
leglen = float(s["rest_lengths"].astype(float).sum(1).mean())
band = a.band * leglen
T = len(ct)
want = set(a.legs.split(",")) if a.legs else None
spec = []
print(f"[[ leg length {leglen:.2f} u | touching band < {band:.2f} u "
      f"({band*9.972:.1f} mm at the measured model scale)")
for l in ALEGS:
    if want and MAP[l] not in want:
        continue
    k = order.index(l)
    z = s[f"toe_{l}"].astype(float)[:, 2] - g
    vel = np.linalg.norm(np.gradient(s[f"toe_{l}"].astype(float), axis=0), axis=1)
    low = z < band
    drag = low & ~ct[:, k]
    runs = []
    for _, gp in itertools.groupby(enumerate(np.where(drag)[0]), lambda t: t[1] - t[0]):
        gg = [x[1] for x in gp]
        runs.append([gg[0], gg[-1]])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= a.gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    merged = [r for r in merged if r[1] - r[0] + 1 >= a.min_len]
    tot = sum(r[1] - r[0] + 1 for r in merged)
    print(f"[[ {MAP[l]}: touching {100*low.mean():4.0f}% | scheduled planted "
          f"{100*ct[:, k].mean():4.0f}% | DRAGGING {100*tot/T:4.0f}% "
          f"({len(merged)} window(s), mean speed while dragging "
          f"{vel[drag].mean() if drag.any() else 0:.2f} u/frame)")
    for r in merged:
        spec.append((r[0], r[1], MAP[l]))
if spec:
    print("\n--force-planted \"" + ";".join(f"{s0},{e0}:{l}" for s0, e0, l in spec) + "\"")
else:
    print("\n[[ no dragging windows at this band")
