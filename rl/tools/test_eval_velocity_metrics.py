"""Offline tests for the eval_velocity metric maths. No Isaac Sim required.

The evaluation battery decides whether Task 1 is complete, so its arithmetic needs
to be right before it is ever pointed at a checkpoint. The subtle part is that a
vec env AUTO-RESETS a terminated env: from the step after a fall, that env's
velocity belongs to a fresh episode and must not be scored against the segment's
command. These tests pin that behaviour.

    python3 rl/tools/test_eval_velocity_metrics.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_recorder():
    """Pull SegmentRecorder out of eval_velocity.py without importing isaaclab."""
    src = open(os.path.join(HERE, "eval_velocity.py")).read()
    start = src.index("class SegmentRecorder")
    end = src.index("def main()")
    ns = {"np": np}
    exec(compile(src[start:end], "eval_velocity.py", "exec"), ns)
    return ns["SegmentRecorder"]


def test_post_termination_excluded():
    R = load_recorder()
    T, E = 20, 4
    r = R("seg", (0.25, 0.0, 0.0))
    for t in range(T):
        done = np.zeros(E, bool)
        if t == 10:
            done[2] = True                      # env 2 terminates here
        v = np.tile(np.array([0.25, 0.0]), (E, 1))
        if t >= 10:
            v[2] = [5.0, 5.0]                   # fresh episode after the auto-reset
        r.add(v, np.zeros(E), 3.0, np.full((E, 12), 0.5), 0.001, 0.0,
              np.zeros((E, 12)), done)
    s = r.summary(3.0)
    assert abs(s["survival"] - 0.75) < 1e-9, s["survival"]
    assert s["falls"] == 1, s["falls"]
    # the post-reset garbage must not pull the mean off the commanded value
    assert abs(s["vx_mean"] - 0.25) < 1e-9, s["vx_mean"]
    assert abs(s["vx_rmse"]) < 1e-9, s["vx_rmse"]
    assert abs(s["tilt_mean_deg"] - 3.0) < 1e-9
    assert abs(s["torque_sat_pct"]) < 1e-9, "0.5 Nm is far below the 3.0 Nm ceiling"
    print("ok  post-termination steps excluded, survival counts any fall in the segment")


def test_tracking_error_and_saturation():
    R = load_recorder()
    T, E = 20, 4
    r = R("seg", (0.25, 0.0, 0.0))
    for _ in range(T):
        r.add(np.tile(np.array([0.20, 0.0]), (E, 1)), np.zeros(E), 2.0,
              np.full((E, 12), 2.99), 0.002, 0.1, np.zeros((E, 12)),
              np.zeros(E, bool))
    s = r.summary(3.0)
    assert s["survival"] == 1.0
    assert abs(s["vx_mean"] - 0.20) < 1e-9
    assert abs(s["vx_rmse"] - 0.05) < 1e-9, s["vx_rmse"]
    assert s["torque_sat_pct"] == 100.0, "2.99 Nm is >= 99% of the 3.0 Nm ceiling"
    print("ok  velocity RMSE and torque-saturation percentage")


def test_action_smoothness():
    R = load_recorder()
    T, E = 10, 2
    r = R("seg", (0.0, 0.0, 0.0))
    for t in range(T):
        act = np.full((E, 12), float(t))        # constant slope: |da| = 1 per step
        r.add(np.zeros((E, 2)), np.zeros(E), 0.0, np.zeros((E, 12)), 0.0, 0.0,
              act, np.zeros(E, bool))
    s = r.summary(3.0)
    assert abs(s["action_rate_rms"] - 1.0) < 1e-9, s["action_rate_rms"]
    print("ok  action-rate RMS")


def test_all_envs_fall():
    """Every env terminating must not produce a crash or a bogus mean."""
    R = load_recorder()
    T, E = 5, 3
    r = R("seg", (0.25, 0.0, 0.0))
    for t in range(T):
        done = np.ones(E, bool) if t == 0 else np.zeros(E, bool)
        r.add(np.zeros((E, 2)), np.zeros(E), 0.0, np.zeros((E, 12)), 0.0, 0.0,
              np.zeros((E, 12)), done)
    s = r.summary(3.0)
    assert s["survival"] == 0.0 and s["falls"] == E
    assert np.isnan(s["vx_mean"]), "no valid steps must report NaN, not 0.0"
    print("ok  total-failure segment reports NaN rather than a fake zero")


if __name__ == "__main__":
    test_post_termination_excluded()
    test_tracking_error_and_saturation()
    test_action_smoothness()
    test_all_envs_fall()
    print("\nall eval-metric tests passed")
