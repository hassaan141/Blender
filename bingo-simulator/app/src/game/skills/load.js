// Build the skill set from motions/index.json.
//
// The index is produced by tools/export_motions.py, which exports ONLY clips with an
// actual Stage-4 pass. Everything else arrives in `not_ready` with the reason, and is
// shown in the UI as unavailable rather than hidden - so the list of what Bingo can
// really do is visible, not implied.

import {MOTION_DIR} from "../constants.js";
import {BingoSkill, RobotMotion, SkillKind} from "./manager.js";

export async function loadSkills() {
  const r = await fetch(`${MOTION_DIR}/index.json`);
  if (!r.ok) throw new Error(`motion index fetch ${r.status}`);
  const idx = await r.json();

  const skills = [];

  for (const entry of idx.ready || []) {
    const mr = await fetch(`./${entry.file}`);
    if (!mr.ok) {
      skills.push(new BingoSkill({
        name: entry.name, kind: SkillKind.GESTURE, ready: false,
        reason: `motion file missing (${mr.status})`,
      }));
      continue;
    }
    const j = await mr.json();
    skills.push(new BingoSkill({
      name: entry.name,
      kind: SkillKind.GESTURE,
      ready: true,
      reason: entry.evidence,
      motion: RobotMotion.fromJson({name: j.name, fps: j.fps,
                                    dof_positions: j.dof_positions,
                                    expression_only: j.expression_only}),
    }));
  }

  for (const n of idx.not_ready || []) {
    skills.push(new BingoSkill({
      name: n.name, kind: SkillKind.GESTURE, ready: false, reason: n.reason,
    }));
  }

  // Locomotion and recovery are declared so the UI can show WHY they are unavailable
  // rather than leaving a silent gap.
  skills.push(new BingoSkill({
    name: "Walk", kind: SkillKind.LOCOMOTION, ready: true, requiresPolicy: true,
    reason: "needs a trained command-conditioned policy",
  }));
  skills.push(new BingoSkill({
    name: "Recover", kind: SkillKind.RECOVERY, ready: false,
    reason: "no get-up policy has been trained for Bingo yet",
  }));

  return skills;
}
