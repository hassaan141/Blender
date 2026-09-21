# BINGO AMP V9 RESTORE + TRAINING HANDOFF

## Goal

Restore the **best historical Bingo AMP locomotion policy, v9**, exactly enough that its known checkpoint can be played again and the result matches the original video.

Do not start another locomotion architecture yet.

The immediate objective is:

```text
restore v9
→ reproduce v9
→ freeze it
→ use it as the baseline for future natural-motion work

The known visual baseline is:

bingo_amp_FINAL_v9.mp4

V9 was not perfectly natural, but it was significantly better than the later AMP work and much closer to what we want.

It had:

clean diagonal trot
tall posture
level body
all four legs stepping
no rear-leg dragging
no bounding
no severe crouching
good forward motion

The remaining issue was that it still looked somewhat mechanical.

1. EXACT CHECKPOINT

Use this exact checkpoint:

/pub0/muhammadf/IsaacLab/logs/skrl/bingo_amp_trot/2026-07-21_20-00-11_amp_torch/checkpoints/agent_48000.pt

DO NOT use:

best_agent.pt

best_agent.pt was a stationary style-max solution and is not the moving v9 gait.

The moving v9 result is:

agent_48000.pt
2. V9 MEASURED RESULT

Historical evaluation:

base height: ~0.192 m
tilt: ~0.008
vx: ~0.568 m/s

Diagonal gait signature:

fl · br = +0.98
fr · bl = +1.00

fl · fr = -0.98
fl · bl = -0.97

This means:

front-left + back-right

move together, while:

front-right + back-left

move together half a cycle later.

This is the desired diagonal trot pattern.

V9 is the historical AMP high-water mark. Later v10-v15 experiments did not beat it visually.

3. IMPORTANT REPO WARNING

The CURRENT files under:

bingo_rl/bingo_rl/amp/

DO NOT represent v9 anymore.

They were later changed during v10-v15, especially v14.

The current code may include things such as:

command-conditioned observations
51-D policy obs instead of 49-D
multi-gait reference
trot + pace
different reward logic
different reference foot representation
different AMP/task weighting

Therefore:

DO NOT simply run the v9 checkpoint through the current play_amp.py.

The checkpoint and environment must match.

Create a SEPARATE restored environment instead.

Suggested location:

bingo_rl/bingo_rl/amp_v9/

Do not overwrite:

bingo_rl/bingo_rl/amp/
4. V9 REFERENCE MOTION

V9 used one retargeted dog trot:

bingo_trot.npz

Original generation was approximately:

python bingo_rl/scripts/retarget_dog_to_bingo.py \
    --dog dog_trot.txt \
    --out motions/bingo_trot.npz \
    --cycles 6 \
    --fwd_speed 0.5

Important retarget constants:

G_SY = 0.15
G_SP = 0.60
G_KN = 0.85

SP0 = -0.3
KN0 = +0.6

The correct bilateral Bingo joint sign mapping must be preserved.

Historical axis/sign finding:

SP signs:
fl +1
bl +1
fr -1
br +1

knee signs:
fl +1
bl +1
fr -1
br -1

Do not assume all four leg joint signs are identical.

5. VERY IMPORTANT: USE THE V9 REFERENCE STYLE

Do NOT use the later v10 shank-tip reference when reproducing v9.

V9 used the simpler knee-hinge foot representation.

That reference was mechanically simplified, but it produced the best policy.

The later naturalness attempt changed the reference to the real shank tip and added large foot lift/body motion.

That became v10 and visually regressed.

For reproduction:

V9 first.
Naturalness later.

Do not mix the two stages.

6. V9 AMP OBSERVATION

V9 used:

49-D AMP observation

Approximately:

12 joint positions
12 joint velocities
1 root height
6 root orientation tangent/normal features
3 root linear velocities
3 root angular velocities
4 feet × 3 positions

Total:

49

The policy controlled:

12 leg DOFs

Head/tail were not part of the leg action policy.

Do NOT use the later 51-D command-conditioned observation.

7. V9 ACTION / POSTURE CONSTRAINT

An important earlier AMP failure was the robot spreading its legs sideways into a very stable but ugly spider stance.

V9 retained the fix:

SY action authority *= 0.3

This prevented excessive shoulder-yaw abduction.

Keep this.

Without it the policy may rediscover the wide crouched cheat.

8. V9 TASK REWARD

The core v9 reward was:

velocity
×
diagonal trot correctness
×
body height
Velocity
vel_rew = torch.clamp(vx / 0.5, 0.0, 1.0)

Standing therefore receives approximately zero velocity reward.

This was important because an exponential velocity reward previously left a non-zero reward floor that allowed AMP to trot in place.

Diagonal trot reward

Contacts:

fl
fr
bl
br

Score:

trot_score = 0.25 * (
    (fl == br)
    + (fr == bl)
    + (fl != fr)
    + (fl != bl)
)

Desired trot:

fl == br
fr == bl
fl != fr
fl != bl

Then:

trot_factor = 0.2 + 0.8 * trot_score

A correct diagonal trot approaches:

1.0

A bound or pace receives much less.

Height reward

Target height:

0.18 m
height_factor = torch.exp(
    -8.0 * torch.clamp(0.18 - base_height, min=0.0)
)

This prevented the policy from satisfying the gait reward by belly-crawling.

Final task reward
task_reward = vel_rew * trot_factor * height_factor

Do not replace this reward during the restoration.

This exact design was the main breakthrough that produced v9.

9. V9 AMP WEIGHTING

Use:

task_reward_scale  = 1.5
style_reward_scale = 1.5

This balance was important.

Earlier tests showed:

too much AMP style
→ stationary / trot-in-place behavior

too much task reward
→ moves, but gait quality degrades

The 1.5 / 1.5 configuration produced the best v9 result.

10. V9 SKRL CONFIG

The historical configuration was approximately:

policy:
  layers: [1024, 512]
  activation: relu

value:
  layers: [1024, 512]
  activation: relu

discriminator:
  layers: [1024, 512]
  activation: relu

Important policy exploration settings:

initial_log_std: -2.9
fixed_log_std: true
entropy_loss_scale: 0.0

Other known values:

rollouts: 16
learning_epochs: 6
mini_batches: 2

discount_factor: 0.99
lambda: 0.95

learning_rate: 5.0e-5

ratio_clip: 0.2
value_clip: 0.2

value_loss_scale: 2.5

discriminator_loss_scale: 5.0
discriminator_gradient_penalty_scale: 5.0
discriminator_logit_regularization_scale: 0.05
discriminator_weight_decay_scale: 1.0e-4

task_reward_scale: 1.5
style_reward_scale: 1.5

DO NOT use the new Locomotion-2 AMP setting:

initial_log_std = -1.0

That belongs to the recent completely different AMP experiment.

For reproducing v9:

initial_log_std = -2.9
11. CONTACT SENSOR

The environment used contact information on the knee/terminal leg bodies.

The historical environment used something equivalent to:

ContactSensorCfg(
    prim_path=".*_knee",
    track_air_time=True,
)

The v9 task reward requires reliable:

fl contact
fr contact
bl contact
br contact

because the diagonal phase reward depends directly on them.

Verify the body/contact ordering.

Do not assume index order without printing the names.

12. TERMINATION / HEIGHT

Historical useful values:

target_height = 0.18
termination_height = 0.15
height_penalty_scale = 8.0
target_speed = 0.5

The low-body termination was introduced because earlier versions discovered a belly-crawl solution.

Keep it for restoration.

13. TRAINING COMMAND

Historical training command:

CUDA_VISIBLE_DEVICES=1 \
./isaaclab.sh -p bingo_rl/scripts/train_amp.py \
    --task Bingo-AMP-Trot-Direct-v0 \
    --algorithm AMP \
    --headless \
    --num_envs 2048 \
    --max_iterations 3000

Approximate training time historically:

~40 minutes

Successful run:

logs/skrl/bingo_amp_trot/2026-07-21_20-00-11_amp_torch/
14. RESTORE STRATEGY

Do NOT revert the whole repository.

Use Git history only to recover the AMP implementation needed for v9.

A useful historical commit around this AMP period is:

b2b1d5b0b9c00012dced02a54fe2122f2c890cf8

However, do NOT blindly replace the repository with that commit.

The documentation in memory.md is the authoritative description of the final v9 recipe because later commits may contain work performed after v9.

Restore only the relevant files into:

bingo_rl/bingo_rl/amp_v9/

Possible files needed:

amp_v9/__init__.py
amp_v9/bingo_amp_env.py
amp_v9/bingo_amp_env_cfg.py
amp_v9/agents/skrl_amp_cfg.yaml

Also create separate scripts/tasks if necessary:

play_amp_v9.py
eval_amp_v9.py
Bingo-AMP-V9-Direct-v0
Bingo-AMP-V9-Direct-Play-v0

Do not modify the current AMP task.

15. FIRST JOB IS PLAYBACK, NOT TRAINING

Before training anything:

Load:

agent_48000.pt

with the restored environment.

Record a close-up video.

The restored result must visually resemble:

bingo_amp_FINAL_v9.mp4

If it does not:

STOP.

Do not retrain.

Find the mismatch first.

Possible mismatch sources:

observation order
joint order
contact body order
action scaling
default pose
SY scale
reference npz
task/environment version
robot asset revision
PD/actuator configuration
normalization/preprocessor state

The checkpoint is only useful when run inside its original environment assumptions.

16. V9 REPRODUCTION EVALUATION

Record at minimum:

vx
base height
roll/pitch
joint trajectories
joint velocity
joint acceleration
foot-tip trajectory
contacts
contact duty
cadence
torque saturation
joint-limit contact

Also calculate gait phase correlations.

Target approximate signature:

fl·br ≈ +0.98
fr·bl ≈ +1.00

fl·fr ≈ -0.98
fl·bl ≈ -0.97

Do not rely only on reward.

WATCH THE VIDEO.

17. FREEZE THE RESTORED V9

Once reproduction passes, create something like:

docs/locomotion_v9/

Store:

RESTORE_REPORT.md
config snapshot
reference metadata
evaluation metrics
video
checkpoint path
Git commit/hash information

The restored v9 system becomes:

READ ONLY

Never overwrite it.

18. WHY WE ARE RESTORING V9

The recent Locomotion-2 AMP run is not better.

That controller achieved forward velocity but produced:

large frantic stride
huge paw excursions
high joint acceleration
~25% torque saturation
joint-limit contact
poor visual naturalness

V9 had already solved the basic gait topology much better.

Therefore we should NOT spend more time trying to make the new AMP run converge toward something we already had.

19. WHAT HAPPENED AFTER V9

Do not repeat the same experiments blindly.

V10

Changed:

knee hinge → real shank tip
added large paw lift
added root height bob
added root pitch/roll

Result:

more movement
but choppy
front legs over-reached
robot lunged forward
diagonal quality degraded

fl·br dropped roughly:

+0.98 → +0.78

The idea that v9 looked mechanical because the reference lacked natural body/foot motion was valid, but the implementation changed too much at once.

V11/V12

Reduced front leg amplitude and body bob.

The lunge improved, but the original clean gait never fully returned.

V13

Forced L/R symmetry in the reference.

The weak leg simply moved somewhere else.

Conclusion:

policy asymmetry can be emergent

and is not necessarily caused by asymmetric mocap.

V14

Large architectural change:

single trot
→ trot + pace

49-D policy
→ 51-D command-conditioned policy

trot reward
→ pure velocity/yaw tracking

task/style
→ more style dominant

It gained command control but visually did not beat v9.

V15

Increased task reward and trot bias.

Regressed further.

Historical conclusion:

V9 remains the best AMP gait.

20. NEXT STAGE AFTER V9 IS RESTORED

DO NOT immediately modify the v9 controller.

First collect a v9 rollout dataset.

Record many gait cycles containing:

joint positions
joint velocities
actions

root position
root orientation
root linear velocity
root angular velocity

foot-tip positions
foot-tip velocities

contacts
contact duration

joint acceleration
torque

This becomes the physically-valid Bingo locomotion dataset.

21. NATURALNESS DIRECTION AFTER RESTORE

The end goal is not just a trot.

The end goal is:

natural command-controlled Bingo locomotion
+
expressive body motion later

We have three useful motion sources:

1. V9
   physically stable Bingo motion

2. online dog mocap
   natural gait structure

3. Blender walk
   Bingo-specific authored motion

The future goal should be to combine their strengths.

Conceptually:

DOG MOCAP
natural gait structure
       \
        \
         → natural-motion target
        /
BLENDER WALK
Bingo-specific visual motion
        \
         \
          → small correction over V9
         /
V9
stable physical locomotion
22. DO NOT DIRECTLY COPY DOG JOINT ANGLES

The dog and Bingo have different morphology.

Natural data should initially provide things like:

contact timing
duty factor
paw trajectory shape
paw clearance
stance compression
cadence
body heave
body pitch
body roll
stride variation

rather than:

dog hip angle
dog knee angle
dog segment orientation

This avoids repeating the previous exact-retarget problem where a kinematically valid dog-derived trajectory was dynamically bad for Bingo.

23. RECOMMENDED NATURALIZATION ARCHITECTURE

Once v9 is frozen:

state
 ↓
V9 policy
 ↓
nominal action
 +
small learned residual
 ↓
Bingo

The residual should initially have limited authority.

Suggested starting point:

10–20% of the normal joint action range

The residual's job is:

subtle paw trajectory improvement
subtle stance compression
subtle body heave
subtle pitch/roll
more organic timing
small stride variation

NOT:

invent locomotion again
24. SUCCESS RULE

Every naturalization candidate is compared directly against v9.

Keep it only if:

video looks more natural than v9
AND
gait remains coordinated
AND
all four legs remain active
AND
velocity remains acceptable
AND
joint acceleration stays reasonable
AND
torque saturation stays reasonable
AND
joint-limit contact stays low

If the metrics improve but the motion looks worse:

REJECT IT

If the robot becomes more animated but choppy:

REJECT IT

If the new approach fails:

V9 remains available unchanged.
25. IMMEDIATE TASK FOR THIS SESSION

Do this autonomously in one substantial pass:

1. inspect current repo
2. inspect memory.md v9 recipe
3. inspect Git history around the July AMP implementation
4. create isolated amp_v9 package
5. reconstruct the exact 49-D v9 environment
6. locate/use the original bingo_trot.npz if available
7. load agent_48000.pt
8. record playback
9. evaluate gait numerically
10. compare against bingo_amp_FINAL_v9.mp4

DO NOT start a new training run until the checkpoint reproduces correctly.

DO NOT modify Locomotion 1.

DO NOT modify the current Locomotion-2 AMP experiment.

DO NOT modify the expressive Stage 1–5 pipeline.

The deliverable for this step is:

A reproducible, frozen AMP V9 teacher controller.

Only after that works should we begin combining:

V9
+ dog mocap
+ Blender walk

to reduce the mechanical appearance.


The important change from the earlier handoff is that I would **not call the next step "train v9" yet**