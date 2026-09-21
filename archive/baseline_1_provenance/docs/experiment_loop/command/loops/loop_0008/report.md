# loop_0008

Loop 7 turns better but develops a leaning stand. At zero command, inherited tracking rewards target a frozen walking pose despite feedforward applying the asset stand pose (0.571 rad RMS mismatch). Match pose, velocity, foot and contact tracking targets to the existing stand blend without changing reward weights, applied references, physics or replay. Recover from unpromoted loop 7; loop 6 remains champion. This is training attempt eight of eight.

REJECT: All ten cases survive and pivot yaw improves to +0.294/-0.272, but stand still has a persistent 17.40-degree lean versus loop6 5.86 degrees. Video confirms an unacceptable stand posture; backward yaw and transitions also remain below goal. Reject final promotion; retain loop6 as the best balanced experimental policy. Eight-attempt bound reached: stop training. The unified merge is functional but the full natural command gait goal is not met.

Full command gate passed: False
