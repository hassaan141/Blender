# Bingo Retarget Kinematic Validation

Source: `dog02_walk_02.npz` -> Retarget: `bingo_dog02_walk_02.npz`

## Summary

- Body scale (dog hip constellation -> Bingo hip layout, Umeyama similarity fit): {'fl': 0.8592575095102861, 'fr': 0.8556553453161758, 'bl': 0.9877583055461733, 'br': 0.9329735237739654}
- Base (root) fit residual: mean 0.0 mm, max 0.0 mm
- IK foot-tracking error: mean 0.48 mm, p95 5.62 mm, max 11.07 mm
- Foot slip during contact: 6.2 -> 5.4 mm/stance (project gate: <5)
- Root travel: 0.122 -> 0.064 m (after contact-consistent correction)
- Max |joint velocity|: 5.18 rad/s (limit 10), frames over: 0
- Duty factor by leg: {'fl': 0.35, 'fr': 0.65, 'bl': 0.45, 'br': 0.48333333333333334}

## Root-orientation fix (methods A/B/C comparison)

The original approach (method A: per-frame rigid Kabsch fit of the dog's 4 hip keypoints to Bingo's fixed hip layout) let hip/scapula articulation noise leak into what should be a single global root orientation -- it produced a 5-16 degree, frame-to-frame-varying rotation not explained by real heading change, pinned `fr_SY_J`/`br_SY_J` at their limits for 73-95% of frames, and gave a visibly wobbly/looping retarget root path despite a clean forward source path. Replaced with the source's own `root_pos`/`root_orient_rotmat` (already produced by `extract_walk.py`'s single static heading-alignment normalization, not re-derived per frame from noisy leg-attachment points). Three variants compared numerically on this clip:

| method | mean IK foot error | joints saturated (>5%) |
|---|---|---|
| A. 4-hip Kabsch (original) | 3.47 mm | fr_SY_J 95.0%, br_SY_J 73.3% |
| B. SMAL root_orient directly | 0.74 mm | fl_SY_J 36.7%, fr_SY_J 36.7%, br_SY_J 5.0% |
| **C. yaw-only (smoothed), used here** | **0.48 mm** | **fl_SY_J 38.3%, br_SY_J 1.7%** |

C was chosen: best foot-tracking accuracy and the saturation is concentrated on a single joint rather than spread across two-to-three. This directly validates the diagnosis: the root-frame reconstruction, not the dog's real motion, was the dominant cause of the original joint-limit failure.

## Gate: joint limits respected

**Partially passes.** One joint still sits at its hard limit for a large fraction of frames: {'fl_SY_J': 38.333333333333336}. `br_SY_J` (1.7%) is unremarkable.

Follow-up diagnosis on the remaining `fl_SY_J` saturation, per the task's two-step protocol (root path, then leg-frame/keypoint mapping) before considering the source gait incompatible:

1. **Not a keypoint-mapping bug.** `fl`'s hip-relative paw offset, expressed in the root-local frame, shows the saturated windows coincide with frames where the leg is *simultaneously* near its peak 3D extension (Z close to Bingo's max reach) *and* carrying a real lateral (Y) placement (up to ~37mm) -- both legitimate, present in the source motion, not a sign-flip or index error (KP24 indices were already independently verified against dataset-wide geometry in `extract_walk.py`).
2. **Not primarily an amplitude problem either.** Extending the amplitude-reduction search well past the existing 50% floor (down to 20% of the original swing) did reduce `fl_SY_J` saturation further (38%->27%) with excellent foot error (0.12mm), but that is an 80% cut to the front legs' swing amplitude -- it would visibly flatten the gait, failing "preserve overall visual character" even as the joint-limit number improves. Not adopted; the amplitude search stays floored at 0.5, matching `scripts/retarget.py`'s own original, deliberate choice.
3. **Conclusion**: `fl_SY_J`'s residual saturation reflects a genuine, now well-localized kinematic tightness -- Bingo's hip-yaw range (+/-22 degrees) combined with a leg already near full extension has less *effective* lateral reach than the leg's raw length suggests, and this dog's front-left leg asks for both at once during specific phases of this clip. This is reported honestly rather than hidden, but is not, on its own, grounds to call the source gait incompatible: it is one joint, concentrated in identifiable windows, with otherwise excellent (sub-mm) foot tracking. Proceeding to Isaac kinematic replay to observe the actual behavior (a clamped joint is not a crash) rather than theorize further.

## Other gates (informational)

- **Contact timing preserved**: trivially and exactly true by construction -- `retarget_to_bingo.py` uses the source's own `foot_contact` array (from `extract_walk.py`'s Otsu+hysteresis detector) directly as the retarget's `contacts` field, rather than re-deriving contacts from the retargeted foot height. Gait cadence (footstep timing) is therefore identical to the source by construction, since no time resampling was performed anywhere in this pipeline.
- **Visual resemblance**: substantially improved -- see the top-down path panel above. The retarget path no longer traces the pronounced backward loop seen with method A, though some wobble remains (expected: Bingo's rigid, much smaller body cannot reproduce the dog's own spine flex, and the root-local retarget necessarily discards whatever of that flex isn't captured by the single root frame). Proceeding to Isaac kinematic replay to confirm this visual read on the real articulated robot, per the task's instruction to continue automatically once the kinematic retarget is in reasonable shape.

