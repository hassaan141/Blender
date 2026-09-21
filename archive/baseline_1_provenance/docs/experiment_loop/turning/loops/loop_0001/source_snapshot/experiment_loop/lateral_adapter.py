"""Lateral (left/right sidestep) campaign hooks for core.py.

Mirrors backward_adapter.py's stage flow (train: geometry -> train; evaluate:
run + gate-check; compare: reuse adapter.comparison()), adapted for a
genuinely new command axis rather than a sign-flipped existing one -- see
docs/experiment_loop/lateral/AUTONOMOUS_LATERAL_PROTOCOL.md.

Unlike backward's per-loop interactive kinematic-approval gate (a separate
polling agent drops KINEMATIC_GATE.json), this campaign's kinematic preview
was reviewed ONCE per direction at campaign start (see
docs/experiment_loop/lateral/{right,left}/kinematic_preview.mp4 +
KINEMATIC_GATE.json) since the SAME agent that runs each loop synchronously
is also the one supervising it -- there is no separate process to poll here.
geometry.py's own IK-feasibility check (raises on failure) is the per-loop
automatic safety net for any geometry-setting changes a later loop proposes.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

EXPERIMENT_LOOP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_LOOP))
from core import ROOT, read, save, digest  # noqa: E402
from adapter import comparison  # noqa: E402  (reused unchanged)

KIT_ARGS = "--/rtx/verifyDriverVersion/enabled=false --no-window"


def direction_of(c: dict) -> str:
    d = c.get("direction")
    if d not in ("right", "left"):
        raise ValueError("config.json must set 'direction': 'right' or 'left'")
    return d


def env_for(c: dict, arm: Path) -> dict:
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(c["gpu"]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "BINGO_LATERAL_REFERENCE": str(arm / "reference.npz"),
        "BINGO_LATERAL_CMD_VY": str(c["settings"]["cmd_vy"]),
    }


def run(cmd, env, log_path: Path | None = None):
    print("EXEC", cmd, flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def stage_train(c: dict, arm: Path):
    direction = direction_of(c)
    env = env_for(c, arm)

    # 1. regenerate the reference fresh from the CUMULATIVE settings dict
    # (same "always re-derive from the fixed source" discipline backward uses).
    geom = subprocess.run(
        [c["python"], str(EXPERIMENT_LOOP / "lateral/geometry.py"), str(arm), "--direction", direction],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True,
    )
    print(geom.stdout, flush=True)

    # 2. train, reusing the generic skrl PPO entrypoint via lateral/worker.py's
    # gym.make() settings-monkeypatch (mirrors adapter.command()'s train argv).
    kit = KIT_ARGS
    cmd = [
        c["python"], str(EXPERIMENT_LOOP / "lateral/worker.py"), "train",
        "--context", str(arm / "config.json"), "--headless",
        "--task", "Bingo-Lateral-Train-v0",
        "--num_envs", str(c["num_envs"]), "--seed", str(c["seed"]),
        "--max_iterations", str(c["iterations"]),
        "--checkpoint", c["training_parent"]["checkpoint"],
        "agent.agent.experiment.directory=" + str(arm / "training"),
        "hydra.run.dir=" + str(arm / "hydra"),
        "--kit_args", kit,
    ]
    save(arm / "train.command.json", {"argv": cmd, "environment": {
        k: env[k] for k in ["CUDA_VISIBLE_DEVICES", "BINGO_LATERAL_REFERENCE", "BINGO_LATERAL_CMD_VY"]
    }})
    run(cmd, env)

    expected = c["iterations"] * 24
    checkpoints = list((arm / "training").glob(f"*/checkpoints/agent_{expected}.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError("Expected final checkpoint missing/ambiguous; exit 0 is insufficient")
    import shutil
    shutil.copy2(checkpoints[0], arm / "policy.pt")
    save(arm / "training_completion.json", {
        "expected_iterations": c["iterations"], "expected_steps": expected,
        "checkpoint": str(checkpoints[0]), "sha256": digest(arm / "policy.pt"),
    })


def stage_evaluate(c: dict, arm: Path):
    direction = direction_of(c)
    env = env_for(c, arm)
    cmd_vy = c["settings"]["cmd_vy"]
    out_dir = arm / "evaluation"
    cmd = [
        c["python"], str(EXPERIMENT_LOOP / "lateral/evaluate.py"),
        "--task", "Bingo-Lateral-Play-v0",
        "--checkpoint", str(arm / "policy.pt"),
        "--cmd_vy", str(cmd_vy),
        "--out_dir", str(out_dir),
        "--duration_s", str(c["duration_s"]),
        "--settle_seconds", "1",
        "--diagnostics", "--headless", "--kit_args", KIT_ARGS,
    ]
    save(arm / "evaluate.command.json", {"argv": cmd})
    run(cmd, env)

    m = read(out_dir / "metrics.json")
    n = read(out_dir / "natural_metrics.json")
    q = read(out_dir / "quality_metrics.json")

    failures = []
    if q.get("warmup_falls", 0) > 0:
        failures.append("warmup fall")
    for leg, v in n["feet"].items():
        if v["duty"] >= 0.85 or (v["clearance_p95_mm"] is not None and v["clearance_p95_mm"] < 5):
            failures.append(leg + " insufficient sustained clearance / high contact duty")
    if not math.isfinite(m.get("forward_drift_abs_max_m_s", float("nan"))) or m["forward_drift_abs_max_m_s"] > 0.12:
        failures.append("excessive forward drift during sidestep")

    # loop-over-loop regression check against the DIRECTION's OWN current
    # champion (there is no pre-existing lateral seed to compare loop 1
    # against, unlike backward's natural_walk/attempt_03 seed).
    champion_metrics_path = Path(c["parent"]["metrics"])
    if champion_metrics_path.exists() and c["parent"]["id"] != "locomotion1_quality_run05":
        cm = read(champion_metrics_path)
        sat = sum(m["torque_saturation_pct_per_joint"].values()) / 12
        csat = sum(cm["torque_saturation_pct_per_joint"].values()) / 12
        if sat > max(csat * 1.5, csat + 5):
            failures.append("mean torque saturation exceeds current lateral best +50%/+5pp")

    save(arm / "lateral_gates.json", {
        "failures": failures,
        "direction": direction,
        "vy_mean": m.get("vy_mean"), "vy_abs_max": m.get("vy_abs_max"),
        "forward_drift_abs_max_m_s": m.get("forward_drift_abs_max_m_s"),
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["train", "evaluate", "compare"])
    p.add_argument("--context", type=Path, required=True)
    a = p.parse_args()
    c = read(a.context)
    arm = Path(c["arm_dir"])
    if not arm.resolve().is_relative_to(ROOT / "docs/experiment_loop"):
        raise ValueError("Output outside experiment_loop")
    if c.get("smoke"):
        raise ValueError("Real hooks cannot run in smoke mode")

    if a.stage == "train":
        stage_train(c, arm)
    elif a.stage == "evaluate":
        stage_evaluate(c, arm)
    else:
        comparison(arm, Path(c["parent"]["video"]), c["duration_s"])


if __name__ == "__main__":
    main()
