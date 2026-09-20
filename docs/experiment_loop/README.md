> Current status (2026-09-18): the authorized autonomous backward campaign has finished at eight total loops. See [BACKWARD_RESULTS.md](BACKWARD_RESULTS.md). Best is loop 7, provisionally retained; the full natural-walk goal remains unmet. Historical infrastructure-only and one-loop sections below describe earlier task scopes.

# Bingo experiment-loop infrastructure

**Built and smoke-tested only. No Isaac Lab training or simulation was launched.**

Code: `rl/bingo_rl/experiment_loop/`. State and all new artifacts: `docs/experiment_loop/`.

[Research and source audit](RESEARCH.md) · [Configuration](config.json) · [Champion](CURRENT_CHAMPION.json) · [Index](LOOP_INDEX.csv) · [Proposal template](proposal.example.json)

Locomotion 1 quality Run 05 remains the champion. Natural-walk attempt 03/reference 02b is configured as the initial **experimental training seed**, with its contact correction enabled. This does not promote that previously rejected candidate. A promoted candidate becomes the next parent. A/B arms always start independently from the same parent, reference, seed and budget.

## Configure a real experiment later

No experiment is queued. Before starting, either add a reviewed proposal JSON path to `plans` in `config.json`, or configure `planner_hook` as an argv array. Do not run the example placeholder as a real hypothesis.

A proposal specifies one `defect`, concrete `evidence` (video timestamps/diagnostics), one `hypothesis`, `expected_effect`, and one or two `arms`. Each arm has a unique name and 1–2 related `changes`. Two arms also require `ab_rationale`. Allowed numeric settings are explicitly bounded in `core.ALLOWED`: residual EMA, residual scale and three existing gait-quality reward weights. Physics, gains, limits and historical source edits are not accepted changes. Extending this allowlist or adding reference-generation/code-edit hooks is a separate reviewable implementation change.

The planner hook receives `{context}` and `{output}` placeholders. It must read the champion, previous report, candidate evidence and optional `FEEDBACK.md`, then write a proposal to the output JSON. It must not launch training or modify existing source/artifacts. A future coding agent can supply this hook; the runner itself does not pretend to diagnose video or call an unconfigured model. With neither a queued plan nor a planner, it fails closed before training. Custom train/evaluate/compare argv hooks can be configured; their contract is the artifacts produced by `adapter.py`. Hooks are trusted local code, not an OS sandbox.

## Start, stop, resume

Run from repository root. The following are **instructions for later use**, not commands executed during setup:

```bash
# One loop, within a total campaign ceiling of 6 loop directories:
PYTHONDONTWRITEBYTECODE=1 python3 rl/bingo_rl/experiment_loop/core.py --run-one --max-loops 6

# Continue until ceiling, a failure, stop request, or pending visual approval:
PYTHONDONTWRITEBYTECODE=1 python3 rl/bingo_rl/experiment_loop/core.py --run-continuous --max-loops 6

# From another terminal: finish the current whole loop (including both A/B arms), then stop:
python3 rl/bingo_rl/experiment_loop/core.py --stop-after-current

# Reconcile state, explicitly clear the stop request, and continue within the same total ceiling:
PYTHONDONTWRITEBYTECODE=1 python3 rl/bingo_rl/experiment_loop/core.py --run-continuous --max-loops 6 --resume
```

`--max-loops` is a cumulative ceiling, including failed loop directories, not extra loops on every resume. At most two trainings per loop. Use a terminal multiplexer for unattended operation; the CLI stays in the foreground. SIGINT stops the current bounded process group and records failure. A runner lock prevents two controllers from sharing state. It only coordinates this runner, not unrelated GPU jobs.

Resume never silently reruns an uncertain training stage. It seals an interrupted directory as FAILED, keeps partial logs/checkpoints, and can proceed to a fresh loop if the explicit budget and next proposal permit. A surviving child holds the runner lock; wait for that bounded job to exit before resuming. A journal reconciles an interrupted champion-pointer update. Completed directories are read-only, SHA-manifested and never reused.

## Visual approval

Metric eligibility means survival 100%, mean vx 0.19–0.30, std <0.06, max <0.40, acceleration ≤120% of the **fixed original Run 05** acceleration, no joint-limit violations, complete finite metrics and full evaluation outputs. Speed is not maximized. A/B ordering after gates is vx std, acceleration, action-rate RMS, torque saturation. Video can reject any eligible candidate.

Eligible loops stop at `PENDING_VISUAL`. Watch the full candidate and comparison videos before recording a decision:

```bash
python3 rl/bingo_rl/experiment_loop/core.py --review loop_0001 --decision KEEP --arm A --note 'Watched full comparison; describe the visible improvement'
python3 rl/bingo_rl/experiment_loop/core.py --review loop_0001 --decision REJECT --arm A --note 'Describe the visible regression'
```

These are alternatives; each loop gets one immutable external review. A human may choose either metric-valid A/B arm. Failed metrics, changed artifacts or stale champion lineage cannot be overridden by approval. Set `require_visual_approval` to `false` only to explicitly disable this boundary; automatic promotion then checks metrics but makes no claim of visually improved naturalness. Default remains **true**.

## Artifacts and reproducibility

```text
docs/experiment_loop/
  README.md, RESEARCH.md, config.json, proposal.example.json
  CURRENT_CHAMPION.json, LOOP_INDEX.csv, PROTECTED_BASELINE.json
  FEEDBACK.md                         # optional human/agent boundary guidance
  loops/loop_0001/
    hypothesis.json, context.json, config_snapshot.json, parent.json
    source_snapshot/, protected_before.json, protected_after.json
    A/                                # optional B/ has same structure
      config.json, changed_files.json, changes.diff, reference.npz
      commands.json, *.command.json, *.log, *.receipt.json
      train.effective_settings.json, evaluate.effective_settings.json
      training/, hydra/, policy.pt, training_completion.json
      evaluation/                     # metrics, quality/natural metrics, paw traces, eval.mp4
      metric_comparison.json, comparison.mp4
    report.md, decision.json, state.json, MANIFEST.json
  reviews/loop_0001.json               # later approval, outside sealed experiment
  smoke/<timestamp>/                  # separate replay-only state; never a live champion
```

Source snapshots include native controller/evaluator and orchestrator. Config diffs represent the bounded changes; existing source is not edited. Exact expanded commands/environment are recorded by each real adapter hook. Physics/control settings are inherited unchanged. The checkpoint/reference pairing and contact correction are carried through promotion.

Protected hashing covers historical docs/results, controller/physics source, Stage assets, motion assets and the prior natural-walk protection inventory. Hash checks run before/after each loop, after planning and real stages, and before promotion. Symlink targets and contents are checked. A mismatch stops work and blocks promotion; it does not attempt destructive automatic restoration. Hashes detect modifications, but do not sandbox arbitrary custom hooks. Python caches and Git metadata are excluded.

Timeouts: planner 300s, training 3600s, evaluation 600s, comparison 300s. Hooks use argv arrays, no shell evaluation, process groups and nonzero-exit checks. A final expected-timestep checkpoint is required in addition to process success; optimizer update counts are not independently audited. Native GPU execution and runtime settings have been statically wired, **not tested with a new simulator launch in this task**.

## No-training validation

```bash
PYTHONDONTWRITEBYTECODE=1 python3 rl/bingo_rl/experiment_loop/test_loop.py
PYTHONDONTWRITEBYTECODE=1 python3 rl/bingo_rl/experiment_loop/core.py --smoke-test
```

Smoke bypasses every training/evaluation/planner hook, copies explicitly labelled historical attempt-03 metrics/checkpoint/video, renders a real two-second champion/candidate comparison using ffmpeg, seals the experiment as REJECT, and checks protected hashes. It never imports Isaac Lab or promotes a candidate. Synthetic unit fixtures exercise approval, failure, timeout and recovery without representing physical results.

## Backward experiment 0001

The separately authorized backward trial uses `backward_config.json` and `backward_proposal.json`, one arm and 300 PPO iterations. The existing controller's negative signed phase reverses the original pose/contact sequence. `backward_analyze.py reference` scales stored joint/body velocities by the same negative phase rate; it does not alter positions, clearance or contacts. `backward_worker.py` fixes both train and play command ranges at −0.20 m/s. No protected source changes are required.

Backward-specific checks use signed mean velocity and maximum **absolute** velocity, plus slip, individual-foot clearance/duty, measured yaw drift and torque-regression checks. `core.py` now recognizes these explicit negative-speed gates. The controller still has no yaw command input: yaw zero is evaluated through actual drift, not claimed as an implemented command channel. The corrected BL contact definition from attempt 03 remains enabled.

The pre-training kinematic review is performed by the agent. Human approval remains required for champion promotion, and cannot override failed metrics. This trial's loop budget is one; do not infer authorization for another loop from its outcome.
