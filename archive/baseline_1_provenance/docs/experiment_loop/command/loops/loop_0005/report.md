# loop_0005

The final mixed sampler loses explicit pivot commands, so PPO undertrains the missing skill. Starting from retained loop 3 and its original reference, use half uniformly sampled nine core commands and half continuous random commands. Preserve reward, replay and reference geometry. Test pivot tracking and stand retention without forgetting teacher gaits.

REJECT: Stand and transitions fall; stand video shows repeated collapse. Balanced command coverage modestly improves left/right pivot yaw to +0.237/-0.126 and reduces rocking, but survival is mandatory. Loop 3 remains champion. The next targeted recovery may use this rejected checkpoint as an unpromoted training parent while repairing stand supervision.

Full command gate passed: False
