// Keyboard input -> intent. No game logic here: this only reports what is held.

export const KEYMAP = {
  KeyW: "fwd", KeyS: "back", KeyA: "yawL", KeyD: "yawR",
  KeyQ: "strafeL", KeyE: "strafeR",
  KeyR: "recover", Space: "reset", KeyP: "push", KeyH: "hud",
  Digit1: "p1", Digit2: "p2", Digit3: "p3",
  Digit4: "p4", Digit5: "p5", Digit6: "p6",
};

export function createKeyboard(target = window) {
  const held = new Set();
  const pressed = new Set();     // edge-triggered, drained each poll
  const onDown = (e) => {
    const a = KEYMAP[e.code];
    if (!a) return;
    if (e.code === "Space") e.preventDefault();
    if (!held.has(a)) pressed.add(a);
    held.add(a);
  };
  const onUp = (e) => {
    const a = KEYMAP[e.code];
    if (a) held.delete(a);
  };
  const onBlur = () => held.clear();
  target.addEventListener("keydown", onDown);
  target.addEventListener("keyup", onUp);
  target.addEventListener("blur", onBlur);

  return {
    isDown: (a) => held.has(a),
    /** Edge events since the last call. */
    drain() { const p = [...pressed]; pressed.clear(); return p; },
    dispose() {
      target.removeEventListener("keydown", onDown);
      target.removeEventListener("keyup", onUp);
      target.removeEventListener("blur", onBlur);
    },
  };
}
