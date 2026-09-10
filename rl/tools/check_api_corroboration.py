"""Static cross-reference: which Isaac Lab names does the new locomotion code rely on,
and which of those are already proven by working Bingo code in this repository?

This is NOT a substitute for rl/tools/verify_locomotion_api.py. That script talks to
the real Isaac Lab and is the only thing that can actually confirm an API. This one
runs anywhere (stdlib only, no numpy, no Isaac) and answers a narrower question:

    of the Isaac Lab names the locomotion module uses, which ones also appear in
    Bingo code that has ALREADY RUN against the local install?

A name used by env_cfg.py, improved_walking_cfg.py, bingo_v4.py, agents.py or the
Stage-5 env is corroborated: that code demonstrably works on the training machine.
A name that appears ONLY in the new locomotion module is an assumption, and is the
part worth reading closely before the first training run.

    python3 rl/tools/check_api_corroboration.py
"""
from __future__ import annotations

import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# Code that has already run against the locally installed Isaac Lab.
REFERENCE_FILES = [
    "rl/bingo_rl/bingo_rl/env_cfg.py",
    "rl/bingo_rl/bingo_rl/improved_walking_cfg.py",
    "rl/bingo_rl/bingo_rl/bingo.py",
    "rl/bingo_rl/bingo_rl/bingo_v4.py",
    "rl/bingo_rl/bingo_rl/agents.py",
    "rl/bingo_rl/bingo_rl/__init__.py",
    "rl/bingo_rl/bingo_rl/stage5/bingo_stage5_env_cfg.py",
    "rl/bingo_rl/bingo_rl/stage5/bingo_stage5_env.py",
    "rl/bingo_rl/bingo_rl/track_v4/bingo_track_v4_env.py",
    "rl/bingo_rl/bingo_rl/track_v4/bingo_track_v4_env_cfg.py",
    "rl/bingo_rl/bingo_rl/track/bingo_track_env.py",
    "rl/bingo_rl/bingo_rl/track/bingo_track_env_cfg.py",
    "rl/tools/track_v4_physics.py",
    "rl/tools/replay_v4.py",
    "rl/tools/eval_stage5.py",
    "stage4/stand_test.py",
]

# The new code under scrutiny.
NEW_FILES = [
    "rl/bingo_rl/bingo_rl/locomotion/bingo_velocity_env_cfg.py",
    "rl/bingo_rl/bingo_rl/locomotion/bingo_velocity_mdp.py",
    "rl/bingo_rl/bingo_rl/locomotion/agents/rsl_rl_ppo_cfg.py",
    "rl/bingo_rl/bingo_rl/locomotion/__init__.py",
    "rl/tools/train_velocity.py",
    "rl/tools/play_velocity.py",
    "rl/tools/eval_velocity.py",
]

# Attribute chains worth tracking: cfg surfaces the env config writes to.
CFG_ROOTS = ("rewards", "events", "commands", "actions", "observations",
             "terminations", "curriculum", "scene", "sim")

# Fields declared by THIS project on its own cfg subclass. They are not Isaac Lab
# API, so listing them as unverified assumptions would be misleading.
OWN_FIELDS = {
    "commands.base_velocity.category_weights",
    "commands.base_velocity.category_ranges",
}

# Names that look like config attributes but are ordinary Python. `self.actions` in
# eval_velocity.SegmentRecorder is a list, so `self.actions.append` is a list method.
NOT_CONFIG = re.compile(r"\.(append|extend|clone|copy|items|keys|values|get)$")


def read(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    return open(p, encoding="utf-8").read()


def collect(rel):
    """Extract the Isaac Lab surface a file touches."""
    src = read(rel)
    if src is None:
        return None
    out = {"imports": set(), "mdp": set(), "cfg": set()}

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"!! {rel} does not parse: {e}")
        return out

    for node in ast.walk(tree):
        # from isaaclab... import X
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(("isaaclab", "isaacsim", "rsl_rl")):
                for a in node.names:
                    out["imports"].add(f"{node.module}.{a.name}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(("isaaclab", "isaacsim", "rsl_rl")):
                    out["imports"].add(a.name)

    # mdp.<name>
    for m in re.finditer(r"\bmdp\.([A-Za-z_][A-Za-z0-9_]*)", src):
        out["mdp"].add(m.group(1))

    # self.<root>.<a>[.<b>] on the config object
    for root in CFG_ROOTS:
        for m in re.finditer(rf"self\.{root}\.([A-Za-z_][A-Za-z0-9_]*)"
                             rf"(?:\.([A-Za-z_][A-Za-z0-9_]*))?", src):
            a, b = m.group(1), m.group(2)
            name = f"{root}.{a}" + (f".{b}" if b else "")
            if not NOT_CONFIG.search(name) and name not in OWN_FIELDS:
                out["cfg"].add(name)
        # also cfg.<root>.<a> (apply_training_stage mutates a passed-in cfg)
        for m in re.finditer(rf"\bcfg\.{root}\.([A-Za-z_][A-Za-z0-9_]*)"
                             rf"(?:\.([A-Za-z_][A-Za-z0-9_]*))?", src):
            a, b = m.group(1), m.group(2)
            name = f"{root}.{a}" + (f".{b}" if b else "")
            if not NOT_CONFIG.search(name) and name not in OWN_FIELDS:
                out["cfg"].add(name)

    # gym entry_point strings name a class without importing it, e.g.
    # entry_point="isaaclab.envs:ManagerBasedRLEnv" in bingo_rl/__init__.py
    for m in re.finditer(r'["\']((?:isaaclab|isaacsim)[\w.]*):(\w+)["\']', src):
        out["imports"].add(f"{m.group(1)}.{m.group(2)}")
    return out


def main():
    ref = {"imports": set(), "mdp": set(), "cfg": set()}
    missing_ref = []
    for rel in REFERENCE_FILES:
        got = collect(rel)
        if got is None:
            missing_ref.append(rel)
            continue
        for k in ref:
            ref[k] |= got[k]

    new = {"imports": set(), "mdp": set(), "cfg": set()}
    per_file = {}
    for rel in NEW_FILES:
        got = collect(rel)
        if got is None:
            print(f"!! new file missing: {rel}")
            continue
        per_file[rel] = got
        for k in new:
            new[k] |= got[k]

    print("=" * 100)
    print("ISAAC LAB API CORROBORATION - static, offline")
    print("=" * 100)
    print("\nThis does NOT verify any API against Isaac Lab. It reports which names the")
    print("new locomotion code shares with Bingo code that already runs on the training")
    print("machine. Run rl/tools/verify_locomotion_api.py there for real verification.\n")
    if missing_ref:
        print(f"note: {len(missing_ref)} reference file(s) not on disk, skipped:")
        for r in missing_ref:
            print(f"      {r}")
        print()

    print(f"reference corpus : {len(REFERENCE_FILES) - len(missing_ref)} files "
          f"({len(ref['imports'])} imports, {len(ref['mdp'])} mdp names, "
          f"{len(ref['cfg'])} cfg attrs)")
    print(f"new code         : {len(per_file)} files "
          f"({len(new['imports'])} imports, {len(new['mdp'])} mdp names, "
          f"{len(new['cfg'])} cfg attrs)")

    total_new = 0
    for kind, label in (("imports", "IMPORTS"), ("mdp", "mdp.* NAMES"),
                        ("cfg", "CONFIG ATTRIBUTES")):
        shared = sorted(new[kind] & ref[kind])
        novel = sorted(new[kind] - ref[kind])
        total_new += len(novel)
        print(f"\n{'-' * 100}\n{label}\n{'-' * 100}")
        print(f"  corroborated by working Bingo code ({len(shared)}):")
        for s in shared:
            print(f"    ok    {s}")
        print(f"\n  NOT found in working Bingo code ({len(novel)}) "
              f"- these are the assumptions:")
        if not novel:
            print("    (none)")
        for s in novel:
            where = [os.path.basename(f) for f, g in per_file.items() if s in g[kind]]
            print(f"    ?     {s:<52s} used in {', '.join(sorted(set(where)))}")

    print(f"\n{'=' * 100}")
    print(f"{total_new} name(s) in the new code are not corroborated by existing Bingo code.")
    print("Each is either a stock Isaac Lab name this project simply never happened to")
    print("use, or a genuine mistake. verify_locomotion_api.py is what tells them apart.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
