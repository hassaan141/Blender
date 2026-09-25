# Attempt 03 hypothesis

Attempts 01 and 02 retained a per-environment command indefinitely unless a fall
occurred. Consequently healthy rollouts did not rotate through the nine command
cases, and the difficult backward-left behavior received too little coverage.
Attempt 03 will end training episodes every 8 seconds and sample backward-left
three times as often as each other command. It starts from frozen BASELINE_1 and
keeps attempt 02's baseline-action penalty (0.5). Success is judged on the same
nine held-command MuJoCo evaluations; the oversampled training case does not alter
the test distribution.
