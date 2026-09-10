// Standard-mapping gamepad. Polled, never event-driven: the Gamepad API has no
// events for axis motion.

const DEAD = 0.15;
const dz = (v) => (Math.abs(v) < DEAD ? 0 : (v - Math.sign(v) * DEAD) / (1 - DEAD));

export function createGamepad() {
  let prevButtons = [];
  return {
    /** null when nothing is connected. */
    poll() {
      const pads = navigator.getGamepads ? navigator.getGamepads() : [];
      const gp = [...pads].find((p) => p && p.connected);
      if (!gp) { prevButtons = []; return null; }
      const b = gp.buttons.map((x) => x.pressed);
      const edge = b.map((v, i) => v && !prevButtons[i]);
      prevButtons = b;
      return {
        id: gp.id,
        // left stick drives the body, right stick aims the head
        moveX: -dz(gp.axes[1] ?? 0),      // up = forward
        moveY: -dz(gp.axes[0] ?? 0),      // left = +y (robot left)
        lookX: dz(gp.axes[2] ?? 0),
        lookY: -dz(gp.axes[3] ?? 0),
        yaw: (gp.buttons[6]?.value ?? 0) - (gp.buttons[7]?.value ?? 0), // triggers
        reset: edge[9] || edge[3],        // start / Y
        push: edge[2],                    // X
        recover: edge[1],                 // B
        nextPersonality: edge[5],         // RB
        prevPersonality: edge[4],         // LB
      };
    },
  };
}
