// Two HUD modes, as the brief asks: a public-facing one and an engineering one.
// The engineering mode is meant to be usable as a controller-debugging tool, so it
// shows commanded vs actual, contacts, torque saturation and joint-limit proximity -
// the things you need to tell "the policy is bad" from "the runtime is wired wrong".

import React from "react";
import {useStore} from "../store.js";
import {JOINT_NAMES, LEG_JOINTS} from "../game/constants.js";
import {PERSONALITIES} from "../game/controllers/expression.js";

const box = {
  position: "absolute", padding: "10px 12px", borderRadius: 8,
  background: "rgba(14,17,21,0.86)", border: "1px solid #232a33",
  backdropFilter: "blur(6px)", pointerEvents: "none",
};

function Row({k, v, warn}) {
  return (
    <div style={{display: "flex", justifyContent: "space-between", gap: 14}}>
      <span style={{color: "#7d8794"}}>{k}</span>
      <span style={{color: warn ? "#e0705f" : "#d7dde4"}}>{v}</span>
    </div>
  );
}

function Bar({label, cmd, act, range}) {
  const pct = (v) => 50 + 50 * Math.max(-1, Math.min(1, v / range));
  return (
    <div style={{marginBottom: 4}}>
      <div style={{display: "flex", justifyContent: "space-between",
                   fontSize: 11, color: "#7d8794"}}>
        <span>{label}</span>
        <span style={{color: "#d7dde4"}}>
          {act.toFixed(2)} <span style={{color: "#5c6673"}}>/ {cmd.toFixed(2)}</span>
        </span>
      </div>
      <div style={{position: "relative", height: 6, background: "#1a1f26",
                   borderRadius: 3, overflow: "hidden"}}>
        <div style={{position: "absolute", left: "50%", top: 0, bottom: 0,
                     width: 1, background: "#39424e"}} />
        <div style={{position: "absolute", left: `${Math.min(pct(cmd), pct(act))}%`,
                     width: `${Math.abs(pct(act) - pct(cmd))}%`, top: 0, bottom: 0,
                     background: "#3d5b7a"}} />
        <div style={{position: "absolute", left: `${pct(act)}%`, top: 0, bottom: 0,
                     width: 2, background: "#6fb3e0"}} />
        <div style={{position: "absolute", left: `${pct(cmd)}%`, top: 0, bottom: 0,
                     width: 2, background: "#e0b45f"}} />
      </div>
    </div>
  );
}

export default function Hud({runtime, skills}) {
  const snap = useStore((s) => s.snapshot);
  const mode = useStore((s) => s.hudMode);
  const toggleHud = useStore((s) => s.toggleHud);
  if (!snap) return null;

  const eng = mode === "engineering";
  const c = snap.contacts || {};
  const dot = (on) => (
    <span style={{display: "inline-block", width: 9, height: 9, borderRadius: 5,
                  marginRight: 4, background: on ? "#4e9e6a" : "#39424e"}} />
  );

  return (
    <>
      {/* status ------------------------------------------------------------- */}
      <div style={{...box, top: 12, left: 12, minWidth: 232}}>
        <div style={{fontWeight: 600, marginBottom: 6, letterSpacing: 0.3}}>BINGO v4</div>
        <Row k="state" v={snap.skillState} />
        <Row k="personality" v={snap.personality} />
        <Row k="policy" v={snap.hasPolicy ? snap.policyName : "NONE"}
             warn={!snap.hasPolicy} />
        {eng && <>
          <Row k="control" v={`${snap.ctrlHz.toFixed(1)} Hz`} />
          <Row k="physics" v={`${snap.physHz.toFixed(0)} Hz`} />
        </>}
      </div>

      {/* no-policy notice --------------------------------------------------- */}
      {!snap.hasPolicy && (
        <div style={{...box, top: 12, left: "50%", transform: "translateX(-50%)",
                     maxWidth: 560, borderColor: "#5a3d33"}}>
          <div style={{color: "#e0b45f", fontWeight: 600, marginBottom: 4}}>
            NO LOCOMOTION POLICY LOADED — Bingo will not walk
          </div>
          <div style={{color: "#9aa4b0", fontSize: 12, whiteSpace: "pre-wrap"}}>
            {snap.policyError}
          </div>
          <div style={{color: "#7d8794", fontSize: 11, marginTop: 6}}>
            Physics, expression and validated gestures still run. Walking is
            deliberately NOT faked with an animation.
          </div>
        </div>
      )}

      {/* velocity tracking -------------------------------------------------- */}
      <div style={{...box, bottom: 12, left: 12, width: 250}}>
        <div style={{fontSize: 11, color: "#7d8794", marginBottom: 6}}>
          COMMAND <span style={{color: "#e0b45f"}}>▮</span> vs ACTUAL{" "}
          <span style={{color: "#6fb3e0"}}>▮</span>
        </div>
        <Bar label="vx (m/s)" cmd={snap.cmd[0]} act={snap.vx ?? 0} range={0.5} />
        <Bar label="yaw (rad/s)" cmd={snap.cmd[1]} act={snap.yawRate ?? 0} range={0.6} />
        <div style={{marginTop: 8, fontSize: 11}}>
          {dot(c.fl)}FL {dot(c.fr)}FR {dot(c.bl)}BL {dot(c.br)}BR
          <span style={{color: "#7d8794", marginLeft: 8}}>
            {snap.contactCount ?? 0}/4 paws
          </span>
        </div>
      </div>

      {/* engineering panel -------------------------------------------------- */}
      {eng && (
        <div style={{...box, top: 12, right: 12, width: 262, maxHeight: "72vh",
                     overflow: "hidden"}}>
          <div style={{fontSize: 11, color: "#7d8794", marginBottom: 6}}>ENGINEERING</div>
          <Row k="base z" v={`${(snap.baseZ * 1000).toFixed(1)} mm`}
               warn={snap.baseZ < 0.14} />
          <Row k="roll / pitch"
               v={`${(snap.roll ?? 0).toFixed(1)}° / ${(snap.pitch ?? 0).toFixed(1)}°`} />
          <Row k="tilt" v={`${(snap.tiltDeg ?? 0).toFixed(1)}°`}
               warn={(snap.tiltDeg ?? 0) > 45} />
          <Row k="max torque" v={`${(snap.maxTorque ?? 0).toFixed(2)} N·m`} />
          <Row k="saturated" v={`${snap.saturated ?? 0} / 21`}
               warn={(snap.saturated ?? 0) > 0} />
          <div style={{marginTop: 8, fontSize: 11, color: "#7d8794"}}>
            LEG JOINTS — pos / target
          </div>
          <div style={{fontSize: 11, marginTop: 3}}>
            {LEG_JOINTS.map((n) => {
              const p = snap.jointPos?.[n] ?? 0;
              const t = snap.jointTarget?.[n] ?? 0;
              const off = Math.abs(p - t) > 0.15;
              return (
                <div key={n} style={{display: "flex", justifyContent: "space-between"}}>
                  <span style={{color: "#5c6673"}}>{n}</span>
                  <span style={{color: off ? "#e0705f" : "#adb6c0"}}>
                    {p.toFixed(3)} / {t.toFixed(3)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* controls ----------------------------------------------------------- */}
      <div style={{...box, bottom: 12, right: 12, width: 250}}>
        <div style={{fontSize: 11, color: "#7d8794", marginBottom: 5}}>CONTROLS</div>
        <div style={{fontSize: 11, color: "#9aa4b0", lineHeight: 1.7}}>
                  <b>W/S</b> forward/back &nbsp; <b>A/D</b> turn<br />
          <b>1-6</b> personality &nbsp; <b>mouse</b> head look<br />
          <b>ctrl+drag</b> orbit &nbsp; <b>ctrl+wheel</b> zoom<br />
          <b>Space</b> reset &nbsp; <b>P</b> push &nbsp; <b>H</b> HUD mode
        </div>
        {snap.lastRefusal && (
          <div style={{marginTop: 7, fontSize: 11, color: "#e0b45f"}}>
            {snap.lastRefusal}
          </div>
        )}
      </div>

      <button
        onClick={toggleHud}
        style={{position: "absolute", top: 12, right: eng ? 286 : 12,
                pointerEvents: "auto", background: "#1a1f26", color: "#9aa4b0",
                border: "1px solid #232a33", borderRadius: 6, padding: "5px 9px",
                cursor: "pointer", fontFamily: "inherit", fontSize: 11}}
      >
        {eng ? "public view" : "engineering"}
      </button>
    </>
  );
}
