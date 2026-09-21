# Baseline gait-quality diagnosis (completed before reward changes)

The instrumented 60-second rollout exactly reproduces retained refinement Run 05's original metrics. Added diagnostics are read-only and retain the original evaluator's definitions. Raw per-step data are saved in baseline/diagnostic_trace.npz.

## Reference asymmetry

The retained reference has 1.081 vs 0.773 rad SP excursion (FL/FR, +40%) and 0.799 vs 0.438 rad knee excursion (+82%). Native-time acceleration RMS is 92.5 vs 70.6 rad/s² at SP and 80.1 vs 68.8 at knee. At actual playback timing the reference acceleration RMS is 25.23 vs 20.15 (SP) and 20.96 vs 18.11 (knee). Rear SP excursions are only 0.661/0.675 rad. Centered, optimally phase/sign-aligned FL/FR waveforms still differ by 0.149 rad SP and 0.168 rad knee: this is not just an opposite-axis-sign or phase-offset artifact. Comparisons of canonical, seam-smoothed and retained references are saved in reference_asymmetry.json. The authored gait is asymmetric; wholesale mirroring would alter the requested gait and is not planned.

## Contact timing and saturation

Using the existing <3 cm shank-tip-height proxy, FL duty is 75.35%, FR 80.28%, BL 100%, BR 86.74%. Reference flags instead give 62.5%, 58.13%, 82.08%, 64.38%. FL knee saturates on 28.11% of proxy-contact samples and 0% of swing samples, versus 7.61%/0% for FR knee. FL SP saturates on 28.73% of swing samples and 13.18% of contact samples, versus FR 17.96%/12.89%.

The proxy is threshold-sensitive: at 1/2/3 cm FL duties are 37.08/62.92/75.35%; FR 36.60/67.57/80.28%. There is no force sensor in this environment, and repeated threshold crossings are not proof of real touchdown impacts. Consequently, FL knee saturation is associated with low-foot support phases, but precise force/load sharing cannot be identified from this proxy alone. The different SP/knee phase patterns argue against one generic saturation cause.

## Residuals and tracking error

FL SP residual RMS is 0.0284 rad vs FR 0.0249; FL knee 0.0378 vs FR 0.0450 (rear knees 0.0404/0.0401). FL knee does not have an unusually large residual. Reference tracking RMS is 0.0340 rad FL knee vs 0.0431 FR, yet tracking the actual residual-adjusted target is worse: 0.0156 vs 0.0085 rad. FL SP adjusted-target error is also larger: 0.0089 vs 0.0054 rad. The residual generally helps follow the desired trajectory; shrinking it indiscriminately could remove compensation. Computed unclipped torque and signed residual/target-error traces are recorded for every joint.

SP/knee actuator gains are identical across legs (Kp 120, Kd 1.60), with the same 3 Nm ceiling. No left/right actuator-setting mismatch explains the difference, and those settings will not change. The strongest evidence is larger authored FL motion plus support-phase target/load mismatch, with residual lag a testable contributor. These are supported hypotheses, not a claim of a uniquely established causal model.

## Body bobbing and fixed cadence

Baseline detrended body-height RMS is 2.3599 mm; height p95–p05 is 8.2759 mm; vertical-velocity RMS is 0.06720 m/s. Phase-cycle period is 2.12191 s (24 Hz sampling). These measures will be collected for every candidate. Global phase rate, reference file, pose sequence, and cycle duration will remain unchanged.

## Initial focused intervention

Test a slightly more responsive residual EMA together with a modest FL SP/knee torque cost. This targets lag and sustained FL drive without changing the reference, gait cadence or actuator physics. Subsequent choices will depend on measured saturation, joint acceleration and bobbing, never increased forward speed.

Measurement frames: body height is the reference body link-origin height; vertical velocity is that body's COM velocity (IsaacLab `body_pos_w` versus `body_lin_vel_w`). They are complementary indicators and are not treated as identical-point derivatives.

## Additional drive-demand check (same baseline trace)

At saturated samples, FL SP computed torque opposes joint velocity about 90.2% of the time (FR 91.0%). FL SP velocity RMS is 1.043 rad/s versus FR 0.836. The unchanged damping term's RMS proxy is 1.669 vs 1.338 Nm, and the spring-error proxy is 1.070 vs 0.651 Nm. This supports high braking/trajectory demand rather than simply insufficient forward push. For the knees, FL/FR velocity RMS is similar (0.530/0.497), while the spring-error proxy differs sharply (1.871/1.018 Nm), consistent with the support-phase tracking/load discrepancy. Post-step PD proxies correlate with computed torque but are not an exact torque decomposition because the samples straddle the last physics integration step. See baseline/pd_demand_diagnostic.json.
