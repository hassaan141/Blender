"""Stitch matched Ashley / v4 stills into one side-by-side sheet.

  python3 stage2/make_compare.py --left <dir> --right <dir> --out sheet.png
"""
import argparse, os
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--left", required=True); ap.add_argument("--right", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--label-left", default="Ashley")
ap.add_argument("--label-right", default="Bingo v4")
ap.add_argument("--by-order", action="store_true",
                help="pair the two directories by sort order rather than by file "
                     "name - needed when one side is a retimed clip and the frame "
                     "numbers no longer line up")
a = ap.parse_args()
fs = sorted(f for f in os.listdir(a.left) if f.endswith(".png"))
gs = sorted(f for f in os.listdir(a.right) if f.endswith(".png"))
pairs = list(zip(fs, gs)) if a.by_order else [(f, f) for f in fs]
rows = []
for f, g in pairs:
    l = Image.open(os.path.join(a.left, f)).convert("RGB")
    rp = os.path.join(a.right, g)
    if not os.path.exists(rp):
        continue
    r = Image.open(rp).convert("RGB")
    h = max(l.height, r.height)
    row = Image.new("RGB", (l.width + r.width, h), (255, 255, 255))
    row.paste(l, (0, 0)); row.paste(r, (l.width, 0))
    d = ImageDraw.Draw(row)
    d.text((6, 6), f"{a.label_left}  {f[1:-4]}", fill=(255, 255, 0))
    d.text((l.width + 6, 6), f"{a.label_right}  {g[1:-4]}", fill=(255, 255, 0))
    rows.append(row)
if not rows:
    raise SystemExit("no matching frames")
out = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), (255, 255, 255))
y = 0
for r in rows:
    out.paste(r, (0, y)); y += r.height
out.save(a.out)
print(f"[[ {len(rows)} matched frames -> {a.out} {out.size}")
