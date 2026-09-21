# loop_0006

Loop 5 improved pivots but catastrophically forgot stand. Original forward-teacher replay includes failed stand trajectories. Replace only those stand demonstrations with complete surviving 20-second loop-3 stand rollouts (75/128 randomized phases survived; 34200 clean states). Preserve moving-teacher data, balanced commands, geometry and reward. Recover from unpromoted loop 5; loop 3 stays champion unless all mandatory retention gates improve.

KEEP: Experimental recovery, not final visual approval. All ten cases survive 60 seconds again; stand has 0% torque saturation and 0.14 rad/s² acceleration. Versus loop 3, right pivot yaw improves from -0.086 to -0.128 with reduced rocking, left pivot rocking falls below 8 degrees, and backward-left acceleration/saturation improve. Teacher-like straight and forward turning gaits remain. Pivot feet still anchor and yaw/transition tracking gates fail, so goal_met remains false.

Full command gate passed: False
