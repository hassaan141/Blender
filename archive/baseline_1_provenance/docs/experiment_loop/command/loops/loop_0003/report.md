# loop_0003

With stable pivot/backward reference geometry established, persistent yaw undershoot is the largest remaining defect. Increase only yaw-tracking reward weight from 0.3 to 0.6, preserving geometry, teacher replay, velocity reward, action interface and all actuator physics.

KEEP: Experimental improvement only: all ten cases survive, yaw improves modestly, backward-left saturation falls from 10.24% to 6.71% and roll RMS falls to 7.8 degrees. Existing walks retained. Pivot rear-left dragging and yaw undershoot persist; NOT final or visually approved.

Full command gate passed: False
