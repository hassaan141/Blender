# Autonomous backward continuation

User authorization: continue from loop 0001 automatically, maximum eight total loops. No human approval between loops. Locomotion 1 remains a protected forward benchmark; `CURRENT_BACKWARD_BEST.json` tracks a separate backward best. Loop 0001 is the authorized starting point, not retrospectively declared a success.

The active agent reads the previous metrics and montages, writes one evidence-backed proposal, inspects kinematic previews, and reviews every completed 60-second evaluation. The file bridge connects these agent decisions to the single continuous runner process. It does not request human input or pre-generate a parameter sweep.

Each new loop warm-starts the retained backward best for 200 PPO iterations, 512 environments, seed 42, command −0.20 m/s. Geometry edits use the original reference 02b plus explicitly cumulative settings and existing FK/IK; original assets remain read-only. Comparison videos show the current backward best and candidate.

KEEP can mean a provisional overall gait improvement while the final target remains unmet. Mandatory retention: full survival, original fixed acceleration cap (+20%), no joint-limit violation, and agent visual judgment of overall improvement without major regressions. Goal success additionally requires all predeclared full acceptance gates and convincing natural stepping. No reward-only promotion is allowed.

The runner stops on convincing success, eight total loops, or an evidence-backed blocked diagnosis after repeated experiments. Stops due to a tool failure are recovered within this same task when possible; an implementation error is not treated as evidence that backward walking is physically blocked.
