"""Integrity helpers for the bounded unified-policy stand-fix campaign."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "docs/experiment_loop/command").is_dir())
HOME = ROOT / "docs/experiment_loop/command/stand_fix"
BASE_PROTECTED = ROOT / "docs/experiment_loop/command/PROTECTED.json"
START = HOME / "START.json"
MAX_LOOPS = 6


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_base() -> dict:
    expected = json.loads(BASE_PROTECTED.read_text())
    bad = []
    for relative, wanted in expected.items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != wanted:
            bad.append(relative)
    if bad:
        raise RuntimeError(f"Protected baseline changed: {bad[:10]}")
    return {"protected_files": len(expected), "unchanged": True}


def initialize() -> dict:
    HOME.mkdir(parents=True, exist_ok=True)
    paths = [
        ROOT / "docs/experiment_loop/command/best/policy.pt",
        ROOT / "docs/experiment_loop/command/best/MANIFEST.json",
        ROOT / "docs/experiment_loop/command/loops/loop_0006/MANIFEST.json",
        ROOT / "docs/experiment_loop/FROZEN_BACKWARD_BASELINE.json",
        ROOT / "docs/experiment_loop/turning/BEST.json",
        ROOT / "docs/walk_ref/quality_runs/run_05/policy.pt",
        ROOT / "rl/bingo_rl/bingo_rl/locomotion/bingo_velocity_env_cfg.py",
        ROOT / "rl/bingo_rl/bingo_rl/bingo_v4.py",
    ]
    record = {
        "max_loops": MAX_LOOPS,
        "parent": "docs/experiment_loop/command/loops/loop_0006",
        "protected": {str(p.relative_to(ROOT)): digest(p) for p in paths},
        "base_check": verify_base(),
    }
    if START.exists() and json.loads(START.read_text()) != record:
        raise RuntimeError("Immutable campaign starting point changed")
    START.write_text(json.dumps(record, indent=2) + "\n")
    return record


def verify_start() -> dict:
    verify_base()
    record = json.loads(START.read_text())
    bad = []
    for relative, wanted in record["protected"].items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != wanted:
            bad.append(relative)
    if bad:
        raise RuntimeError(f"Campaign starting point changed: {bad}")
    return {"starting_files": len(record["protected"]), "unchanged": True}


def seal(loop: int) -> dict:
    if not 1 <= loop <= MAX_LOOPS:
        raise ValueError("loop outside bounded campaign")
    verify_start()
    directory = HOME / "loops" / f"loop_{loop:04d}"
    if not (directory / "decision.json").is_file():
        raise RuntimeError("decision.json required before sealing")
    manifest_path = directory / "MANIFEST.json"
    if manifest_path.exists():
        raise RuntimeError("loop already sealed")
    manifest = {
        str(path.relative_to(directory)): digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json"
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"loop": loop, "files": len(manifest), "sealed": True}


def verify_all() -> dict:
    verify_start()
    results = []
    for manifest_path in sorted((HOME / "loops").glob("loop_*/MANIFEST.json")):
        directory = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        actual = {
            str(path.relative_to(directory)): digest(path)
            for path in directory.rglob("*")
            if path.is_file() and path.name != "MANIFEST.json"
        }
        if actual != manifest:
            raise RuntimeError(f"sealed loop changed: {directory}")
        results.append(directory.name)
    return {"verified_loops": results, "count": len(results)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "check", "seal", "verify"])
    parser.add_argument("--loop", type=int)
    args = parser.parse_args()
    if args.command == "init": result = initialize()
    elif args.command == "check": result = verify_start()
    elif args.command == "seal": result = seal(args.loop)
    else: result = verify_all()
    print(json.dumps(result, indent=2))
