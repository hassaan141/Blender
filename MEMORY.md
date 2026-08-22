# Bingo Animation → RL Pipeline — Working Memory

Last updated **2026-08-21** (paths refreshed for the reorganised folder; findings unchanged since 2026-08-05). Everything here was verified by running it, not inferred.

---

## 1. What exists

Two independent paths from Blender to an Isaac Lab motion `.npz` (spec A.7 schema).

### Path A — legacy art rig (`blend/legacy_art_rig/BingoRig_Latest.blend` and the 6 performance files in `blend/legacy_art_rig/`)

| Script | Does |
|---|---|
| `scripts/bake_motion.py` | Blender → world-space foot/hip/root positions as JSON, 120 Hz. No rig surgery. |
| `scripts/retarget.py` | NumPy: hip-relative scaling → damped-least-squares IK against the URDF → contact-consistent root → A.7 `.npz` |
| `scripts/replay_motion.py` | Isaac Sim kinematic replay of a `.npz` on the robot → PNG frames |

Needed because the art rig is **not** the robot. Lossy: amplitude had to be cut to 50–70%.

### Path B — conform rig (preferred, new)

| Script | Does |
|---|---|
| `scripts/build_rig.py` | URDF → `blend/conform/Bingo_ConformRig.blend`. Correct by construction. |
| `scripts/check_rig.py` | 20 PASS/FAIL checks of the rig against the URDF |
| `scripts/test_ik.py` | 33-case IK acceptance test (tracking / hinge legality / limits) |
| `scripts/bake_conform.py` | animated conform rig → A.7 `.npz` **directly, no retargeting** |

**Verified exact:** baked joint angles pushed through the URDF's own FK reproduce Blender's
foot positions to **0.0005 mm**.

---

## 2. Robot facts (all read from `bingo_urdf_rev_3_real_values.urdf`)

- 17 joints, floating base `origin`. 3 per leg: `SY` (sideways) → `SP` (fore/aft) → `knee`.
  **No ankle, paw, toe, or spine joint.** Contact point is the shank tip, 0.120 m below the knee.
- Mass **2.4781 kg**. Limits: legs 3.0 N·m / 10 rad/s, head+tail 1.5 N·m / 8 rad/s.
- `SY` ±0.42 rad; `SP` and `knee` ±1.56–1.57 rad (±89°, ~179° total travel).
- Thigh 0.0836 m, shank 0.120 m, wheelbase 0.1181 m.

**Non-obvious, cost real time to find:**

- **The paw mesh hangs 28–30 mm below the shank tip.** Rest the tip on z=0 and the paw is
  buried. This also reconciles "stance 0.19–0.20 m" with FK-to-tip giving 0.157 m.
- **The spec contradicts itself on the crouch.** A.3 says SP −0.30 / knee +0.60, but that
  yields 0.1859 m against A.1's stated 0.19–0.20 m. **−0.25 / +0.50 → 0.1989 m** and is what
  the rig ships with.
- **Left/right axis conventions are inconsistent** (`fr_SP_J`, `fr_knee`, `br_knee` flipped;
  `+SY` is inward on the front legs, outward on the back). Building FK straight from the URDF
  axes sidesteps the whole issue — never hand-apply the sign map.
- **Some joint origins carry non-zero `rpy`.** A joint's axis lives in its *parent link's*
  frame, so it must be rotated into world before comparing. Forgetting this produced a
  30° false alarm that took several rounds to unwind.
- **Effort/velocity limits are probably placeholders** — identical round numbers on every
  joint, and `head_pitch`/`head_yaw`/`tail_pitch` have **mass = 0** while `head_roll` carries
  0.79 kg. Needs the mech team's real motor data. No torque feasibility check exists yet.

---

## 3. Foot reach — the biggest authoring constraint

The standing pose puts each paw **ahead of its own hip** (front +103 mm, back +44 mm), so
almost all forward travel is already spent. Measured by IK; Blender's solver and an
independent NumPy solver agree exactly, so this is the robot, not a bug.

| Stance | Forward | Back |
|---|---|---|
| 0.210 m | 16 mm | 169 mm |
| **0.199 m (default)** | **35 mm** | **188 mm** |
| 0.190 m | 48 mm | 201 mm |
| 0.180 m | 60 mm | 213 mm |

Rule for animators: **sweep paws backward, not forward.** Lower the body for more forward room.

Do **not** re-centre the neutral pose to put feet under the hips — a valid such pose exists,
but `bingo_trot.npz` uses the forward-splayed stance, so changing it would desync the rig
from the trained policy.

---

## 4. Conform rig design decisions (each one was a bug first)

- **Bone local Z is the motor axis.** Set the roll numerically, then verify from
  *object-mode* `bone.matrix_local`. `align_roll()` fails silently on zero-offset joints,
  `EditBone.z_axis` reads stale right after `.roll` is assigned, and `EditBone.matrix` does
  not stick. The generator's own self-check lied for several rounds.
- **Foot controls are NOT parented to `root`.** Parenting drags planted paws along with the
  body and bakes in foot sliding — the exact artefact Path A spends effort removing.
- **IK chain = 4** (`ik_end` + knee + SP + SY) with `lock_ik_x/y` on all three joints. Two
  DOF against a 3-DOF target is unsolvable and the solver silently does nothing. Unlocking
  the axes "works" but yields off-hinge poses the robot cannot reach — worse, because it
  fails silently.
- `ik_end` must stay fully IK-locked; unlocking it makes tracking worse.
- SY gets `ik_stiffness_z = 0.95` but **no hard cap** — capping it to the policy's authority
  (±0.126 rad) blew foot drift from 3 mm to 29 mm. `bake_conform.py` warns instead.

---

## 5. Clip verdict (Path A, the 6 delivered performances)

Only **DeadPan** retargets cleanly: 0.01 mm IK error, 1.4% clamping, root 0.192 m.
**Laidback fails** — crouches below the knee limit, 35% of frames clamped.
Cheeky/Enthusiastic are bounds with flight phases. Eccentric is reared, hind feet never contact.

These are an **expressive layer, not a locomotion dataset.** Gait still needs the spec's B.2 capture.

---

## 6. Still open

1. **Schema is 12 DOF.** Head/tail are exported (`head_tail_positions`) but ignored until it
   is extended to 17. Without this the personality work never reaches the robot.
2. **No torque feasibility check** — position and velocity only. Blocked on real motor data.
3. **SY overrun**: policy scales SY ×0.3 (≈±0.126 rad) but IK spends up to 0.30 rad holding
   feet planted. Warned, not solved.
4. **Training side unchanged** — AMP still has no tracking objective, phase input, or early
   termination (spec B.4).
5. Contact flags are inferred from paw height + speed. Real per-foot flags from the animator
   would make the root solve exact.

---

## 7. Environment gotchas (kiwi)

- **`./isaaclab.sh -p` does not work here** — it wants `_isaac_sim/python.sh`, but this is the
  pip install of Isaac Sim 5.1 inside the `isaaclab` conda env. Use plain `python` after
  `source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab`.
  The repo's `record_*.sh` scripts are broken for this reason.
- **RTX rendering hangs without a flag.** Kit misparses the driver (actual 535.288.01,
  reported as 535.32) and its version check fails, so the renderer never initialises and any
  camera render blocks forever at zero frames with no traceback. Always pass:
  `--kit_args "--/rtx/verifyDriverVersion/enabled=false --no-window"`
- **No H.264 encoder** — no `libx264`, no `libopenh264`. Encode `mpeg4` here, re-encode
  locally with `ffmpeg -c:v libx264 -profile:v baseline -pix_fmt yuv420p -movflags +faststart`.
  VS Code will not play High-profile or MPEG-4 Part 2.
- **`git push` needs `--no-verify`** until `git-lfs` is installed (LFS pre-push hook aborts).
- **Never put `pkill -f <pattern>` in the same ssh command as the thing it matches** — the
  command line contains the pattern, so it kills itself and returns silently. This masked a
  working fix for several rounds.

---

## 8. Handoff to the animator

Send **`blend/conform/Bingo_ConformRig.blend`** + **`docs/Bingo_Robot_Spec_for_Animation.md`**.
(`.blend1` is Blender's stale auto-backup — do not send it.)

She returns the animated `.blend`; run `scripts/check_rig.py` on it, then `scripts/bake_conform.py`.
No export step, no format choice, no unit conversion on her side.

Her existing constraint work on the old Rigify rig is superseded — tell her to switch before
investing more. The old rig is a Rigify character rig built around the art model; it was never
derived from the URDF, which is why Path A is lossy.
