# loop_0002

Correct the command-to-reference geometry: cancel common fore-aft stride for pivots, keep backward turns at the retained backward cadence, and solve conservative yaw deformation on the backward posture itself. This should restore backward stepping and near-zero translation without changing rewards or actuator physics. Loop 1 is an unpromoted learning seed, not an approved champion.

KEEP: KEEP as experimental learning baseline only: all ten cases survive, pivot translation is near zero, backward turns now move backward, hard limits remain valid and saturation improves. Yaw undershoot, rocking and backward speed variation remain; NOT a final command policy and visual goal not passed.

Full command gate passed: False
