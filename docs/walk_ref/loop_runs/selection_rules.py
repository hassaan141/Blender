"""Selection guards anchored to the immutable original measurement."""
import math


def protocol_rejections(m, original):
    reasons = []
    keys = ('survival', 'vx_mean', 'vx_std', 'vx_max', 'joint_accel_rms', 'residual_action_rate_rms')
    if any(not math.isfinite(m[k]) for k in keys):
        return ['non-finite metrics']
    if m['survival'] != 1.0 or m['any_fell']:
        reasons.append('survival must be 100%')
    if m['joint_accel_rms'] > 1.2 * original['joint_accel_rms']:
        reasons.append('acceleration exceeds 120% of original baseline')
    if m['reference_tracking_error_rad']['rms'] > 1.25 * original['reference_tracking_error_rad']['rms']:
        reasons.append('reference tracking guard failed')
    return reasons


def old_selection(m, parent, original):
    reasons = protocol_rejections(m, original)
    if m['vx_max'] > max(.4, parent['vx_max'] + .02):
        reasons.append('peak-speed guard failed')
    def speed_pass(x):
        return .2 <= x['vx_mean'] <= .3 and x['vx_std'] < .06
    if not speed_pass(parent):
        if math.hypot(m['vx_mean']-.25, m['vx_std']) >= math.hypot(parent['vx_mean']-.25, parent['vx_std']):
            reasons.append('velocity RMSE did not improve')
    elif not speed_pass(m):
        reasons.append('speed gates regressed')
    elif parent['vx_max'] >= .4:
        if m['vx_max'] >= parent['vx_max']:
            reasons.append('peak did not improve')
    elif m['vx_max'] >= .4 or m['joint_accel_rms'] >= parent['joint_accel_rms'] or m['residual_action_rate_rms'] > parent['residual_action_rate_rms']:
        reasons.append('speed/shaking requirements not improved')
    return reasons


def refinement_selection(m, parent, original):
    reasons = protocol_rejections(m, original)
    if m['vx_max'] >= .4:
        reasons.append('peak must be below 0.40 m/s')
    # A higher mean cannot purchase extra shaking, even under the absolute cap.
    if m['vx_mean'] > parent['vx_mean'] and (m['joint_accel_rms'] > parent['joint_accel_rms'] or m['vx_std'] > parent['vx_std']):
        reasons.append('faster but shakier')
    def rank(x):
        distance = max(.2-x['vx_mean'], 0., x['vx_mean']-.3)
        sat = x['torque_saturation_pct_per_joint']
        # Once all speed gates pass, acceleration comes before further std gains.
        std_violation = max(x['vx_std']-.06, 0.)
        return (distance, x['vx_std'] >= .06, std_violation,
                x['joint_accel_rms'], x['residual_action_rate_rms'],
                sat['fl_SP_J']+sat['fl_knee'], sum(sat.values())/len(sat))
    if rank(m) >= rank(parent):
        reasons.append('no improvement under refinement priorities')
    return reasons
