# Natural forward walk: bounded reference-first loop

Start from immutable Locomotion 1 quality Run 05. All outputs stay under docs/natural_walk; isolated task/scripts under rl/bingo_rl/bingo_rl/natural_walk. Inherit the existing residual PPO environment and validated physics without modification. No broad AMP tuning. Maximum three training attempts that fail to produce a visual improvement.

Compare the Bingo/Blender reference and InterPet source in task space. Preserve the original slow cadence and gait family. A candidate must pass a kinematic visual check before training; then physics-test with the retained controller and, separately, zero residual. Failed direct PD playback is recorded separately from learned-controller performance. No copied dog angles or new actuator settings.

Train from the same canonical checkpoint, 512 environments, seed 42, 100 PPO iterations (2400 steps), command band 0.20–0.30 m/s, original rewards. Reference is the experimental variable. Evaluate for 60 seconds after one-second settling at held command 0.25 m/s, same camera and original metric definitions. Reference previews are kinematic, not survival evidence.

Survival mandatory; target actual speed approximately 0.20–0.30 (baseline measures 0.193). Do not purchase speed with shakiness. Keep acceleration <=120% of canonical 11.00019 rad/s² as a rejection guard; report all saturation/slip/contact/body/limit metrics and flag regressions. Contact and slip use collision geometry and material-point velocity, not invented contact-force measurements. Cadence report distinguishes full reference-loop period from per-leg step events.

VIDEO IS THE HARD GATE: inspect sampled close-up sequences across multiple cycles and full-run overview; reject obvious shuffle/lunge/hop/crouch/jitter or worse visual quality regardless of reward/metrics. Do not claim success for imperceptible changes. Record visual uncertainty honestly and supply the matched comparison video for the user's final judgment.

After two rejected attempts, attempt 3 makes one evidence-driven exception to unchanged rewards: replace only the BL quarter of the existing .10 contact-match term with collision-hull clearance <3mm rather than animation-tip height <3cm. Its weight, other three feet, reference02b, PPO settings and all physics remain unchanged. This addresses a measured contact-label mismatch; it is not broad reward tuning.
