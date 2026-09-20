# Cleanup of unused sidestepping attempt

Removed the Claude lateral/sidestep campaign, failed run outputs, separate implementation, adapter, handoff, and unused lateral-only runner branches. Turning had reused the lazy Isaac Lab registration pattern; that implementation remains in `turning/worker.py`. No retained controller imports the removed package.

Forward, backward, and turning runs remain unchanged, including sealed source snapshots. Rejected runs in those campaigns retain useful optimization history.

The live turning protection guard was migrated only for the deliberately removed lateral paths. Historical per-loop hashes and reports still describe the files present when training ran. [Cleanup accounting](CLEANUP.json).
