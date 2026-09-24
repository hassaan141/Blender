// Wires the game core to React. All the logic lives under src/game/; this file only
// boots the runtime, pumps input into it, and renders.

import React, {useCallback, useEffect, useRef, useState} from "react";

import GameCanvas from "./scene/GameCanvas.jsx";
import Hud from "./ui/Hud.jsx";
import SkillBar from "./ui/SkillBar.jsx";
import {useStore} from "./store.js";
import {BingoRuntime} from "./game/runtime/sim.js";
import {createController} from "./game/controls/controller.js";
import {loadSkills} from "./game/skills/load.js";
import {PERSONALITIES} from "./game/controllers/expression.js";

const PERSONALITY_KEYS = Object.keys(PERSONALITIES);

export default function App() {
  const [runtime, setRuntime] = useState(null);
  const ready = useStore((s) => s.ready);
  const error = useStore((s) => s.error);
  const setReady = useStore((s) => s.setReady);
  const setError = useStore((s) => s.setError);
  const toggleHud = useStore((s) => s.toggleHud);
  const ctrlRef = useRef(null);
  const detachMouse = useRef(null);

  // ---- boot ----------------------------------------------------------------
  useEffect(() => {
    let rt = null;
    (async () => {
      try {
        const skills = await loadSkills();
        rt = await new BingoRuntime().init({skills});
        rt.start();
        setRuntime(rt);
        setReady(true);
      } catch (e) {
        console.error(e);
        setError(e.message || String(e));
      }
    })();
    return () => { rt?.stop(); };
  }, [setReady, setError]);

  // ---- input pump ----------------------------------------------------------
  useEffect(() => {
    if (!runtime) return;
    const ctrl = createController();
    ctrlRef.current = ctrl;
    let raf = 0;

    const pump = () => {
      const s = ctrl.sample();
      runtime.setCommand(s.cmd[0], s.cmd[1]);
      runtime.expr.setLook(s.look[0], s.look[1]);

      // A non-zero command REQUESTS walking; the skill manager decides if it may.
      const moving = Math.hypot(s.cmd[0], s.cmd[1]) > 1e-3;
      if (moving) runtime.skills.requestWalk(runtime.hasPolicy);
      else runtime.skills.requestStand();

      for (const e of s.events) {
        if (e === "reset") runtime.reset();
        else if (e === "push") runtime.push();
        else if (e === "hud") toggleHud();
        else if (e === "nextPersonality" || e === "prevPersonality") {
          const i = PERSONALITY_KEYS.indexOf(runtime.expr.personality);
          const d = e === "nextPersonality" ? 1 : -1;
          const n = (i + d + PERSONALITY_KEYS.length) % PERSONALITY_KEYS.length;
          runtime.expr.setPersonality(PERSONALITY_KEYS[n]);
        } else if (/^p[1-6]$/.test(e)) {
          const n = PERSONALITY_KEYS[Number(e.slice(1))];   // p1 -> index 1
          if (n) runtime.expr.setPersonality(n);
        }
      }
      raf = requestAnimationFrame(pump);
    };
    raf = requestAnimationFrame(pump);
    return () => { cancelAnimationFrame(raf); ctrl.dispose(); };
  }, [runtime, toggleHud]);

  const onCanvas = useCallback((el) => {
    detachMouse.current?.();
    if (el && ctrlRef.current) detachMouse.current = ctrlRef.current.attachMouse(el);
  }, []);

  if (error) {
    return (
      <div style={{padding: 24, maxWidth: 720}}>
        <h2 style={{color: "#e0705f"}}>Simulator failed to start</h2>
        <pre style={{whiteSpace: "pre-wrap", color: "#9aa4b0"}}>{error}</pre>
        <p style={{color: "#7d8794"}}>
          MuJoCo WASM and onnxruntime-web load from jsdelivr, so this needs network
          access. Everything else is served locally.
        </p>
      </div>
    );
  }

  return (
    <div style={{position: "relative", width: "100%", height: "100%"}}>
      <GameCanvas runtime={runtime} onCanvas={onCanvas} />
      {!ready && (
        <div style={{position: "absolute", inset: 0, display: "grid",
                     placeItems: "center", color: "#7d8794"}}>
          loading MuJoCo, robot model and policies…
        </div>
      )}
      {ready && <Hud runtime={runtime} />}
      {ready && <SkillBar runtime={runtime} />}
    </div>
  );
}
