// Merge every input source into ONE command the runtime consumes. Keeping this
// separate means the runtime never learns whether a key or a stick produced a value.

import {VEL_FWD, VEL_BACK, VEL_LAT, VEL_YAW} from "../constants.js";
import {createKeyboard} from "./keyboard.js";
import {createGamepad} from "./gamepad.js";

export function createController(target = window) {
  const kb = createKeyboard(target);
  const pad = createGamepad();
  const look = {x: 0, y: 0};

  // Mouse look while the pointer is over the canvas.
  const onMove = (e) => {
    const r = e.currentTarget?.getBoundingClientRect?.();
    if (!r) return;
    look.x = ((e.clientX - r.left) / r.width) * 2 - 1;
    look.y = -(((e.clientY - r.top) / r.height) * 2 - 1);
  };

  return {
    attachMouse(el) { el.addEventListener("mousemove", onMove); return () => el.removeEventListener("mousemove", onMove); },
    /** {cmd:[vx,vy,yaw], look:[x,y], events:[...]} */
    sample() {
      const g = pad.poll();
      let vx = 0, vy = 0, wz = 0;
      if (kb.isDown("fwd")) vx += VEL_FWD;
      if (kb.isDown("back")) vx += VEL_BACK;
      if (kb.isDown("strafeL")) vy += VEL_LAT;
      if (kb.isDown("strafeR")) vy -= VEL_LAT;
      if (kb.isDown("yawL")) wz += VEL_YAW;
      if (kb.isDown("yawR")) wz -= VEL_YAW;

      if (g) {
        vx += g.moveX * (g.moveX > 0 ? VEL_FWD : -VEL_BACK);
        vy += g.moveY * VEL_LAT;
        wz += g.yaw * VEL_YAW;
      }
      const events = kb.drain();
      if (g?.reset) events.push("reset");
      if (g?.push) events.push("push");
      if (g?.recover) events.push("recover");
      if (g?.nextPersonality) events.push("nextPersonality");
      if (g?.prevPersonality) events.push("prevPersonality");

      return {
        cmd: [
          Math.max(VEL_BACK, Math.min(VEL_FWD, vx)),
          Math.max(-VEL_LAT, Math.min(VEL_LAT, vy)),
          Math.max(-VEL_YAW, Math.min(VEL_YAW, wz)),
        ],
        look: [g ? g.lookX : look.x, g ? g.lookY : look.y],
        events,
        gamepad: g?.id ?? null,
      };
    },
    dispose() { kb.dispose(); },
  };
}
