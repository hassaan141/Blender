# Stage 2 Cheeky failure diagnosis (pre-fix)

This diagnosis was recorded before changing the Stage 2 solver.

## 1. Why the walk became sliding/shifting

The source-to-target axis conversion was incomplete in the leg target code. Ashley
positions were reflected once with `F`, and the solver correctly derived an alignment
matrix `A` from Ashley's anatomical axes to the v4 axes. Root position and orientation
used `A`, but the source hip-to-paw vectors did not:

```python
# failing code
R_root.T @ source_hip_to_paw

# required code
R_root.T @ A @ source_hip_to_paw
```

The old robot feet fit the omitted-`A` targets at 48.3 mm mean error, but are 87.8 mm
from the correctly transformed Ashley trajectories. The evaluator contained the same
omission, so the previous 69.0 mm headline metric did not measure the intended common
frame consistently.

The contact schedule itself is copied exactly from the source and its detected stance
samples are genuinely stationary (Ashley mean horizontal stance speed is only
1.0-1.6 mm/s per foot). The target feet move 69-104 mm/s during those same stance
samples. The post-IK root de-slip integrates frame increments and then smooths the
cumulative correction by six frames; it therefore neither enforces the anchors nor
preserves the source root path. Final root travel is 1.149 m versus Ashley's correctly
aligned 1.378 m, a 229 mm deficit. Swing-path distance is also 13-16% lower than the
source. Global two-sample joint smoothing (83 ms) contributes attenuation, but is not
the primary cause. The largest correctly measured gait mismatches are frames 155, 156,
129, 130, 128, and 154; the largest joint changes cluster at frames 155-160.

## 2. Why the left ear was wrong

Stage 2 sampled `def_Ear.L/R`, but those bones do not carry the complete visible ear
pose. The visible ear meshes are bone-parented to the child animator bones
`Anim_Ear.L/R`, whose large local rotations were omitted. The source file also has
crossed mesh labels: object `ear.L` is parented to `Anim_Ear.R`, and object `ear.R` to
`Anim_Ear.L`. The anatomical bone mapping after the single required X reflection is
still `Anim_Ear.L -> v4 left` and `Anim_Ear.R -> v4 right`; mesh object names must not
be used to swap the channels a second time.

There is a second chain-space error: the old solver matched each ear's orientation
relative to the body while solving only the two ear joints, even though the physical
v4 ear inherits the already-solved three-joint head chain. This double-counts head
motion in the ear joints. The fix must use the visible `Anim_Ear.*` terminal transforms
and solve each ear relative to the achieved v4 head orientation and its own v4 rest
frame.

## 3. What the floating bones are

They are not source-rig remnants or detached mesh objects. The final file contains one
armature, `Bingo_Robot`. Its physical root is animated away from animator controls that
remain at rest, leaving those controls visibly behind the moving robot. The visible
nonphysical set is ten `ctrl_*` bones, four `*_ik_end` bones, and four `*_foot_tip`
markers. Playback already round-trips using only the physical `root` plus the 21 v4
joints. The final bake now deletes these 18 nonphysical bones, leaving the unchanged
physical `root` plus 21-joint skeleton.
