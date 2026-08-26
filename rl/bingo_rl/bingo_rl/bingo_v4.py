"""Articulation config for the v4 Bingo (21 joints incl. ears), from the converted v4 USD."""
import copy
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .improved_walking_cfg import BINGO_IMPROVED_CFG

from pathlib import Path
V4_USD_PATH = str(Path(__file__).resolve().parents[2] / "v4_usd/bingo_v4.usd")  # Blender/rl/v4_usd

BINGO_V4_CFG = copy.deepcopy(BINGO_IMPROVED_CFG)
BINGO_V4_CFG.spawn.usd_path = V4_USD_PATH
BINGO_V4_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.22),
    joint_pos={
        ".*_SY_J": 0.0,
        "fl_SP_J": -0.3, "bl_SP_J": -0.3, "fr_SP_J": +0.3, "br_SP_J": -0.3,
        "fl_knee": +0.6, "bl_knee": +0.6, "fr_knee": -0.6, "br_knee": -0.6,
        "head_.*": 0.0, "tail_.*": 0.0, ".*_ear_.*": 0.0,
    },
)
# actuators: legs (measured rev_1 gains carried in BINGO_IMPROVED_CFG), head/tail, and ears.
BINGO_V4_CFG.actuators = copy.deepcopy(BINGO_IMPROVED_CFG.actuators)
BINGO_V4_CFG.actuators["ears"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_ear_.*"],
    effort_limit_sim=1.0,
    velocity_limit_sim=8.0,
    stiffness=0.6,
    damping=0.05,
)

# ---------------------------------------------------------------- Stage 4 gains
# Derived, not eyeballed: stage4/actuator_analysis.py runs full-tree URDF FK with
# the real v4 masses over the Eccentric reference poses and measures, per joint,
# the static gravity torque tau_g and the effective inertia I about its own axis.
#
#   Kp = max|tau_g| / 0.05 rad      (worst-case steady-state droop <= 0.05 rad)
#   Kd = 2 * sqrt(Kp * I)           (critically damped at that Kp)
#
# Measured max|tau_g| (N m): head_pitch 0.487, head_yaw 0.267, SP 0.21-0.26,
# knee 0.11, SY 0.04-0.12, tail 0.045, ears 0.004.
#
# EFFORT LIMITS ARE DELIBERATELY LEFT UNCHANGED. Statics needs only ~0.5 N m, so
# sizing effort from gravity alone (2x tau_g -> 0.25-0.97 N m) would REDUCE
# authority and leave nothing for contact and dynamics. The rev_1 limits (legs
# 3.0, head/tail 1.5, ears 1.0 N m) are the motor spec and already give ample
# headroom; the baseline's torque saturation came from huge tracking error, not
# from a gravity load the motors cannot meet.
#
# STIFFNESS was the real deficiency: at the old Kp the worst-case gravity droop
# was head_pitch 0.325 rad, head_yaw 0.178, SP 0.115-0.144, SY 0.097-0.103.
# The values below bring every one of those under 0.05 rad.
# CORRECTION: the first pass sized Kp from the FREE-HANGING subtree torque, which
# is the wrong load case for standing. When a paw is on the ground the ground
# reaction acts at the contact point, giving knee/SP ~0.84 N m (not 0.11), measured
# by stage4/actuator_analysis-style statics with the reaction included. Kp = tau/0.05
# then needs ~17 at SP and knee, ~4 at SY. Kd = 2*sqrt(Kp*I) with the URDF subtree
# inertias (SY 4.0e-3, SP 4.0e-3, knee 8.9e-4 kg m^2).
# Iteration 3: statics needs only 0.51 N m (measured with ground reaction at the
# solved stance) against a 3.0 N m ceiling, so the collapse is a TRANSIENT, not a
# static overload: at Kp 17 the drive only reaches its 3.0 N m ceiling once the
# error has already grown to 0.18 rad, by which point the leg is buckling. Raising
# Kp makes it resist at ~0.05 rad instead, well inside the effort budget.
BINGO_V4_CFG.actuators["legs"].stiffness = {".*_SY_J": 40.0, ".*_SP_J": 120.0, ".*_knee": 120.0}
# Iteration 15 - the old Kd was sized to critically damp each JOINT in isolation
# (Kd = 2*sqrt(Kp*I) with I the URDF subtree inertia + armature). That is the wrong
# mode. Under load the leg is not a free joint: it is a spring holding the whole
# 2.45 kg base against the floor, and the mode that actually goes unstable is base
# roll/pitch on leg compliance, whose effective inertia (m*h^2 ~ 0.02 kg m^2 about
# the contact line) is >10x the joint's. Critically damping THAT needs ~3.5x the
# joint-level Kd. Measured on Timid: fall frame 31 -> 39, first joint divergence
# frame 12 -> 26, mean joint error 0.0290 -> 0.0242 rad. Sweep: Kd(SP/knee) 1.4/0.65
# -> 2.5/2.29 -> 3.0 -> 4.0 -> 6.0 gave fall frame 30/31/31/39/31, i.e. a clear
# optimum at 4.0 (6.0 is overdamped and lags).
# CAPPED BY THE STANDING GATE, not by Timid. At Kd(SP/knee) 2.20 stand_test fails on
# "sustained torque saturation" (fl/fr_SP_J, fr_knee pegged at 3.0 N m for >50% of the
# final second) and at 4.00 all eight SP/knee joints are pegged: solver-level velocity
# noise x a large Kd is enough to hold the drive at its ceiling while merely standing.
# 1.60 is the largest value that still passes all eight stand_test criteria (leg
# tracking 0.0041 rad, tilt 1.72 deg, 4/4 contacts, zero sustained saturation) and it
# is still 2.5x the old knee Kd. On Timid it moves mean joint error 0.0290 -> 0.0277
# rad and first joint divergence frame 12 -> 24; the fall frame is unchanged (see the
# note below on why the fall is not a gain problem at all).
BINGO_V4_CFG.actuators["legs"].damping = {".*_SY_J": 0.90, ".*_SP_J": 1.60, ".*_knee": 1.60}
# Iteration 14 - the legs have the SAME defect as the head chain, just milder.
# Proof: with the effort ceiling verified at 15 N m the drive delivers 15.000 N m
# and the joint STILL does not move - byte-identical result to 3.0 N m. A 15 N m
# drive on a 2.46 kg robot that cannot move a joint is blocked, not torque-limited.
# Cause: *_shoulder_yaw is only 14-16 g sitting between the 302 g torso and the
# 160 g thigh (a ~20:1 mass ratio), the same near-massless intermediate-link
# conditioning failure that armature just fixed for head/tail/ears.
BINGO_V4_CFG.actuators["legs"].armature = 0.01
# Iteration 8 - THE actual blocker. A fixed-base test showed the legs hold the
# stance to 0.008 rad with zero leg-torque saturation, while head_pitch/yaw/roll
# were all pinned AT THEIR LIMITS with the drive saturated at 1.5 N m. head_roll
# carries 0.708 kg (29% of the robot); once it flops the CoM shifts far enough to
# topple the floating base, and the legs are then driven into their limits by the
# falling body. That is why raising leg Kp (17->60) and leg effort (3->12) changed
# nothing. Size the head drive to actually hold its own mass.
BINGO_V4_CFG.actuators["head_tail"].effort_limit_sim = 6.0
BINGO_V4_CFG.actuators["head_tail"].stiffness = {"head_.*": 60.0, "tail_.*": 8.0}
BINGO_V4_CFG.actuators["head_tail"].damping = {"head_.*": 1.80, "tail_.*": 0.20}
# Iteration 9 - the real defect. Every joint in the head/tail/ear chains sits AT its
# limit no matter the torque (head_pitch 0.65 rad at 6 N m, tail 0.60 rad at 4.8 N m)
# while the legs track to 0.008 rad. Those chains all run through a near-massless
# intermediate link: head_yaw is 5 g driving the 0.708 kg head - a 140:1 mass ratio
# the articulation solver cannot condition, so the joint jams instead of moving.
# Armature is the standard remedy and is physically real here: these are geared
# servos, whose reflected rotor inertia (N^2 * I_rotor) genuinely dominates the
# bracket's own inertia.
BINGO_V4_CFG.actuators["head_tail"].armature = {"head_.*": 0.06, "tail_.*": 0.02}
BINGO_V4_CFG.actuators["ears"].armature = 0.0005
# Iteration 15 - armature audit. These values are NOT derivable from the URDF: the
# measured subtree inertias about each axis (stage4/actuator_analysis.py) are SY
# 2.2-3.3e-3, SP 2.5-3.5e-3, knee 8.9e-4, head_pitch 1.31e-2, head_yaw 5.2e-3,
# head_roll 1.44e-3, tail 3.4e-4, ear 1.5e-5 kg m^2, so the armature above inflates
# effective inertia by 3x (legs) to 60x (tail). No servo datasheet exists in the
# repo, so reflected rotor inertia cannot be computed and the values stand purely on
# what the solver needs. That was tested, one group at a time, on Timid:
#   legs 0.01 -> 0.002       : mean joint err 0.0290 -> 0.0403 rad, slip 245 -> 501 mm
#   head 0.06 -> 0.01        : fall frame 31 -> 29
#   all groups to ~subtree I : mean joint err 0.0290 -> 0.1396, MAX 2.01 rad, i.e. the
#                              near-massless-intermediate-link jam returns outright
# Every reduction regresses, so the current values are kept. They are a numerical
# conditioning fix with an empirical justification, not a measured physical parameter -
# treat them as provisional until real gearbox specs are available.
# Ears keep Kp 0.6 (tau_g is only 0.004 N m). Their old instability was the
# 1e-12 kg intermediate links, fixed in the URDF, not a gain problem. Kd 0.05 is
# above the critical 0.008 for the new 2 g rotor, i.e. safely overdamped.

# ------------------------------------------------- Stage 4: explicit PD actuators
# THE root cause of the standing failure. With ImplicitActuatorCfg, IsaacLab writes
# Kp/Kd to the PhysX drive but the force ceiling stays at the USD's authored
# maxForce (3.4e38), so effort_limit is never enforced - it only scales the
# *reported* applied_torque. Proof: leg effort 3.0 / 8.0 / 15.0 N m produced
# byte-identical motion (same joint angles, same root z to 4 decimals) while
# reporting +-3.000 / +-8.000 / +-15.000. The robot sank 55 mm and every leg joint
# sat at ~0.13 rad error that no gain could remove.
#
# IdealPDActuator computes tau = clip(Kp*e - Kd*qd, +-effort) in python and applies
# it as a real joint effort, so both the gains and the ceiling actually bite.
# Same test, only this changed: leg tracking 0.1317 -> 0.0041 rad, root sink
# 55 mm -> 4 mm, torso clearance 58 -> 108 mm.
from isaaclab.actuators import IdealPDActuatorCfg as _PD

def _to_explicit(name):
    a = BINGO_V4_CFG.actuators[name]
    BINGO_V4_CFG.actuators[name] = _PD(
        joint_names_expr=a.joint_names_expr,
        effort_limit=getattr(a, "effort_limit_sim", None) or getattr(a, "effort_limit", 3.0),
        velocity_limit=getattr(a, "velocity_limit_sim", None) or getattr(a, "velocity_limit", 10.0),
        stiffness=copy.deepcopy(a.stiffness),
        damping=copy.deepcopy(a.damping),
        armature=copy.deepcopy(getattr(a, "armature", 0.0)),
    )

for _g in ("legs", "head_tail", "ears"):
    _to_explicit(_g)


# ---------------------------------------------------------------- Timid, iteration 15
# WHY THE TIMID FALL IS NOT AN ACTUATOR PROBLEM. Exhaustively ruled out on the
# controller and simulation side (all measured, Timid, fall frame 31 baseline):
#   qdot_ref feed-forward .............. fall 31 -> 30 (regresses; see track_v4_physics)
#   24 Hz target stair-stepping ........ already first-order-held to 120 Hz by default
#   leg Kd 0.65 .. 6.0 ................. best case fall 39, and >1.60 breaks standing
#   armature, per group ................ every reduction regresses (note above)
#   ground friction 0.05 .. 2.0 (40x) .. fall 31 vs 32, i.e. no effect
#   physics dt 1/120, 1/240, 1/480 ..... fall 30/29/29, fully converged
# And the poses themselves are sound: held statically under full physics, reference
# frames 0/10/16/20/24/30 each hold for 5 s with joint error <=0.003 rad, 4/4 paws
# down and <=2.8 deg tilt. Statics is not the problem; the transitions are.
# stage4/dynamic_audit.py finds the actual cause - the reference is 95% STATICALLY
# stable but only 90% DYNAMICALLY feasible, and the ZMP leaves the support polygon
# by up to 314 mm exactly over frames 8, 11, 14-16, 25 and 27-36. The tracker's first
# joint divergence is frame 24 and the fall is frame 30: inside that window. No PD
# gain can hold a base whose required centre of pressure is outside the feet.
