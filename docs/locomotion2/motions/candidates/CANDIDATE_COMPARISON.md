# Locomotion 2 Candidate Comparison (4 backup clips)

Generated with `docs/locomotion2/dataset/evaluate_candidates.py`, reusing
`extract_walk.py`'s keypoint identification, contact detection, and
normalization. This evaluates `dog02_walk_01`'s four intended backups after
`dog02_walk_01` itself (`interpet_dog02_p05_take04_ego_001`) was rejected for
being accelerate-cruise-decelerate rather than sustained steady walking.
No retargeting or training was performed, and no canonical NPZ was written
for any candidate here -- this is evaluation only.

## Headline finding

**None of the four candidate clips contain a window that satisfies
straightness > 0.9 together with >= 4 gait cycles, no stop, and plausible
contact duty factors at the same time.** An exhaustive brute-force scan
(thousands of candidate (start, length) windows per clip) found this
combination is simply not present. Worse, **the automated numeric scoring
was actively misleading for three of the four clips**: `dog01`, `dog06`, and
`dog10`'s best-scoring windows by the numeric criteria (low speed CV,
consistent cycle counts, plausible duty factors) turned out, on rendering
and looking at the validation plot, to be the dog standing and fidgeting or
turning in place -- net displacement of a few centimeters over 3-5 seconds,
forward speed oscillating through zero -- not walking. `dog06`'s window
additionally contains two simultaneous four-paw height crashes (a real
tracking glitch, not gait). Only `dog01_p01_take10`'s and
`dog02_p05_take06`'s windows show genuine (if imperfect) forward locomotion,
and of those two only `dog02_p05_take06` is visually a clean, undistorted
walking arc. This is exactly the kind of failure mode the operator (headless
SSH, cannot view images) cannot catch from numbers alone -- see
`extract_walk.py`'s and `dog02_walk_01`'s development history in memory for
the earlier instances of this same lesson.

## Methodology

For each clip: brute-force scan windows from 2.5-6.5s (0.15-0.2s start step)
against hard sanity gates (median per-leg contact-segment count >= 3, no
frame with smoothed speed under 30% of the window mean, contact duty factor
between 0.15 and 0.85 on every leg, mean speed under 0.5 m/s to exclude
running), then pick the gate-passing window that jointly maximizes
straightness and per-leg footfall-count consistency, using duration as a
tiebreaker. Root-relative (not ground-relative) paw height is used for
contact detection throughout, per the fix documented in `dog02_walk_01`'s
extraction. Every picked window's validation plot (top-down path, paw
heights + contacts, forward speed) was rendered and visually reviewed before
drawing conclusions -- this reversed the automated ranking (below).

## Per-candidate results

| clip | window (s) | dur | cycles (per-leg) | straightness | speed mean±std | duty | visual verdict |
|---|---|---|---|---|---|---|---|
| `interpet_dog01_p01_take10_ego_001` | [9.60, 12.35] | 2.75s | 4.0 `[4,4,3,6]` | 0.363 | 0.152±0.027 | [0.33,0.41,0.51,0.30] | **Not walking.** Top-down path is a small zigzag scribble, ~0.15m total span. Forward speed oscillates -0.02 to +0.16 with no sustained direction -- reads as weight-shifting/fidgeting, not locomotion. |
| `interpet_dog02_p05_take06_ego_001` | **[0.60, 2.60]** (refined from [0.60,3.10] by trimming a wobbly near-stop tail) | 2.00s | 3.0 `[3,3,3,3]` | **0.764** | 0.170±0.125 | [0.35,0.45,0.65,0.48] | **Genuine walk.** Smooth continuous forward arc (0 to ~0.35m), lateral wander under 2.5cm, double-hump accelerate/cruise/decelerate forward-speed profile (0 -> 0.31 -> 0.23 -> 0.33 -> 0), contacts alternate sensibly across the window. The only clip of the four that is unambiguously "sustained forward walking" on inspection. |
| `interpet_dog06_p14_take02_ego_001` | [3.90, 7.65] | 3.75s | 5.0 `[5,5,5,4]` | 0.188 | 0.118±0.033 | [0.18,0.27,0.16,0.35] | **Not walking, and has a tracking glitch.** Top-down path is a chaotic scribble covering ~0.07x0.09m total. Forward speed oscillates -0.06 to +0.18 through zero repeatedly. All four paw heights crash simultaneously to near-zero/negative twice (~t=1.7s, ~t=2.3s) -- a real fit instability, not a footfall. Scored highest on the automated composite (0.694) purely because low-amplitude fidgeting produces low speed CV and, coincidentally, consistent small-scale "footfall" counts. |
| `interpet_dog10_p09_take01_ego_001` | [2.40, 5.90] | 3.50s | 4.0 `[4,3,4,4]` | 0.055 | 0.096±0.064 | [0.53,0.62,0.32,0.18] | **Not walking.** Top-down path is a tangled loop, ~0.09x0.03m total span. Forward speed swings from -0.16 to +0.14, crossing zero repeatedly -- turning/circling in place, not translating. Straightness of 0.055 (essentially a closed loop) makes this the weakest candidate. |

Tracking-quality and morphology numbers for all four (dropped-frame ratio,
mean SMAL `kp_weight`, mean `pet_npy` confidence, body length, withers
height, leg lengths, dog size rank) are in `candidate_metrics.json`.
`dog02`'s morphology in this window: body length 0.614m, withers height
0.409m, leg lengths 0.156-0.200m (size rank #6 of 12, mid-pack -- see
[[bingo-locomotion2-interpet4d]]-equivalent `DATASET_REPORT.md` Section 6).

## Why the automated composite ranking was wrong, and the corrected ranking

The composite score (weighted: duration, speed CV, cycle count, contact
quality, straightness, tracking quality, morphology, per the task's stated
priority order) initially ranked `dog06` first (0.694), then `dog01` (0.467),
`dog02` (0.428 -- on the *untrimmed* window), then `dog10` (0.427). Every one
of those numeric ingredients is computable without ever looking at the
motion, and every one of them was satisfied by "stand still and jitter
slightly" at least as well as by real walking -- low CV and consistent tiny
"footfall" counts don't require net displacement. Straightness was the one
metric in the composite that *did* correctly identify the failure (`dog06`
0.188, `dog10` 0.055, `dog01` 0.363, all low), but it was weighted 5th out
of 7 per the requested priority order, so it wasn't enough to overrule the
others on its own.

**Corrected ranking, incorporating the visual finding that three of the four
are not walking:**

1. **`interpet_dog02_p05_take06_ego_001`** -- only clip with genuine sustained forward walking.
2. `interpet_dog01_p01_take10_ego_001` -- some real (if minimal) forward progress, but mostly fidgeting; a distant second.
3. `interpet_dog10_p09_take01_ego_001` -- no net translation, circling in place.
4. `interpet_dog06_p14_take02_ego_001` -- no net translation, plus an actual tracking glitch; disqualified.

## Final recommendation: `interpet_dog02_p05_take06_ego_001`

**Use `interpet_dog02_p05_take06_ego_001`, window [0.60, 2.60] (2.0s, 60
frames), as the next Locomotion 2 candidate to extract into a canonical
NPZ**, pending your confirmation -- no NPZ was written for it here, only the
validation plot and metrics above.

Why: it is the only one of the four backups where the dog is actually
walking forward. Its numbers are still short of the original target
(straightness 0.764 vs the requested >0.9; 3 cycles vs >=4; forward speed
ranges 0 to 0.33 m/s rather than a flat 0.15-0.35 m/s), and it shares
`dog02_walk_01`'s underlying issue of being an accelerate-then-decelerate
bout rather than a flat cruise -- this appears to be a structural property of
this egocentric, natural-interaction dataset (real dogs in short unscripted
clips move in bouts, not treadmill-constant strides), not something a
better window-search would fix. But "short of target, honestly reported" is
a categorically different and better outcome than the other three
candidates' "numerically fine, visually not walking at all." Do not
substitute `dog01`, `dog06`, or `dog10` for `dog02_p05_take06` on the
strength of any single automated metric without re-rendering and looking at
the plot first.

No retargeting or training was performed. The original dataset files were
not modified.
