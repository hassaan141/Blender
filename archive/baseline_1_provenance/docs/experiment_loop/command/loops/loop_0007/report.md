# loop_0007

Pivot yaw undertracks with near-zero translation. Increase pivot-only yaw-bank geometry input by 1.5, interpolating/clamping within the existing validated +/-0.6 reference bank. Preserve cadence, moving-turn geometry, clean stand replay and rewards. Short preflight survives all commands and improves pivot yaw to +0.256/-0.166; test full stability, rocking and planting after PPO.

REJECT: All cases survive and pivot yaw improves to +0.271/-0.228, but stand develops a persistent 18.79-degree lean and forward saturation rises to 18.57%, failing retention. Video confirms the lean. Loop 6 remains champion. The final recovery attempt addresses the measured inconsistency between standing feedforward and frozen walking reward targets, without changing applied reference geometry or actuator physics.

Full command gate passed: False
