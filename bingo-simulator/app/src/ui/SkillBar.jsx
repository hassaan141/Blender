// Skill and personality buttons. Unavailable skills are SHOWN, disabled, with the
// reason on hover - the brief asks for a clear list of what is genuinely validated,
// and hiding the rest would hide exactly that.

import React from "react";
import {useStore} from "../store.js";
import {PERSONALITIES} from "../game/controllers/expression.js";
import {SkillKind} from "../game/skills/manager.js";

const btn = (on) => ({
  padding: "5px 10px", borderRadius: 6, cursor: on ? "pointer" : "not-allowed",
  background: on ? "#1d2630" : "#15191e",
  border: `1px solid ${on ? "#2f3d4c" : "#20262d"}`,
  color: on ? "#cfd8e2" : "#4e565f",
  font: "inherit", fontSize: 11,
});

export default function SkillBar({runtime}) {
  const snap = useStore((s) => s.snapshot);
  const [, force] = React.useReducer((x) => x + 1, 0);
  if (!runtime || !snap) return null;

  const gestures = runtime.skills.list(SkillKind.GESTURE);
  const persona = Object.values(PERSONALITIES);

  return (
    <div style={{position: "absolute", bottom: 12, left: "50%",
                 transform: "translateX(-50%)", display: "flex", gap: 16,
                 alignItems: "flex-start", pointerEvents: "auto"}}>
      <div style={{padding: "9px 11px", borderRadius: 8,
                   background: "rgba(14,17,21,0.86)", border: "1px solid #232a33"}}>
        <div style={{fontSize: 11, color: "#7d8794", marginBottom: 6}}>
          EXPRESSION — head · tail · ears
          <span style={{color: "#5c6673"}}>  (not gait — see README)</span>
        </div>
        <div style={{display: "flex", gap: 6, flexWrap: "wrap", maxWidth: 420}}>
          {persona.map((p) => (
            <button
              key={p.label}
              style={{...btn(true),
                      outline: snap.personality === p.label ? "1px solid #6fb3e0" : "none"}}
              title={p.measured
                ? `measured RMS rate — tail ${p.rms.tail}, head ${p.rms.head} rad/s`
                : "neutral baseline"}
              onClick={() => { runtime.expr.setPersonality(p.label); force(); }}
            >{p.label}</button>
          ))}
        </div>
      </div>

      <div style={{padding: "9px 11px", borderRadius: 8,
                   background: "rgba(14,17,21,0.86)", border: "1px solid #232a33"}}>
        <div style={{fontSize: 11, color: "#7d8794", marginBottom: 6}}>
          GESTURES — Stage-4 validated only
        </div>
        <div style={{display: "flex", gap: 6, flexWrap: "wrap", maxWidth: 340}}>
          {gestures.map((s) => (
            <button
              key={s.name}
              style={btn(s.ready)}
              disabled={!s.ready}
              title={s.reason}
              onClick={() => { runtime.skills.request(s.name); force(); }}
            >{s.name}</button>
          ))}
        </div>
      </div>
    </div>
  );
}
