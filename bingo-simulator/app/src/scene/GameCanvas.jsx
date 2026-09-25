// The R3F layer. It owns the camera, lights and ground, and drives the render rig
// from the runtime's physics state once per frame.
//
// The physics loop is NOT here. It runs on its own fixed-step clock in
// game/runtime/sim.js; this component only reads the result. That separation is why
// a dropped frame cannot change the simulation.

import React, {useEffect, useRef, useState} from "react";
import {Canvas, useFrame, useThree} from "@react-three/fiber";
import * as THREE from "three";

import {buildRenderRig, loadKinematics} from "./renderRig.js";
import {useStore} from "../store.js";

function Rig({runtime}) {
  const {scene} = useThree();
  const rigRef = useRef(null);
  const setSnapshot = useStore((s) => s.setSnapshot);
  const acc = useRef(0);

  useEffect(() => {
    let alive = true;
    (async () => {
      const kin = await loadKinematics();
      const rig = await buildRenderRig(kin);
      if (!alive) return;
      rigRef.current = rig;
      scene.add(rig.placer);
      // Debug handle. Exposed deliberately: it is how tools/check_rig_vs_physics.mjs
      // verifies that the rendered robot agrees with MuJoCo rather than drifting into
      // a decorative animation, which is the one failure this design must not have.
      if (typeof window !== "undefined") {
        window.__bingo = {...(window.__bingo || {}), rig, THREE};
      }
    })();
    return () => {
      alive = false;
      if (rigRef.current) scene.remove(rigRef.current.placer);
    };
  }, [scene]);


  useFrame((_, dt) => {
    const rig = rigRef.current;
    if (!rig || !runtime?.sim) return;
    if (typeof window !== "undefined" && !window.__bingo?.runtime) {
      window.__bingo = {...(window.__bingo || {}), runtime};
    }
    // Guarded: the MuJoCo/ORT Embind glue can throw intermittently
    // ("_emval_take_value has unknown type memory_view<bool>"). An uncaught throw
    // here escapes the R3F frame loop and corrupts React's scheduler
    // ("Should not already be working"), which kills the root and the HUD with it.
    try {
      const snap = runtime.snapshot();
      rig.setFromPhysics(snap.basePos, runtime.sim.baseQuat(), snap.jointPos);
    } catch { /* skip this frame */ }
    // The HUD does not need 120 updates a second.
    // NOTE: do NOT call setSnapshot() here. Pushing React state from inside the
    // R3F frame loop can re-enter React's scheduler ("Should not already be
    // working"), which kills the root and takes the HUD with it. The snapshot is
    // published from a plain interval below instead.
  });
  return null;
}

// Third-person chase camera (Microduck-style): before Start it holds a fixed
// overview; after Start it sits BEHIND Bingo along the robot's own heading and
// follows. The old camera used a fixed WORLD offset, so once Bingo turned, the
// camera ended up in front of it looking at its face.
const CAM_DIST = 1.05;     // metres behind the robot
const CAM_HEIGHT = 0.68;   // metres above the look-at point (elevated 3rd-person)
const CAM_IDLE = new THREE.Vector3(0.85, 0.45, 0.85);

const CAM_TRANSITION_MS = 1400;   // ease from the idle angle into the chase pose

// easeInOutCubic
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

// Ctrl + drag orbits the camera around Bingo, Ctrl + wheel changes distance. These
// are OFFSETS applied to the chase pose, not a free-fly camera, so the camera still
// follows and still sits behind the robot - the user just chooses where "behind" is.
function useCamOrbit() {
  const {gl} = useThree();
  const off = useRef({az: 0, el: 0, dist: 0});
  useEffect(() => {
    const el = gl.domElement;
    let drag = false, lx = 0, ly = 0;
    const down = (e) => { if (!e.ctrlKey) return; drag = true; lx = e.clientX; ly = e.clientY; e.preventDefault(); };
    const move = (e) => {
      if (!drag) return;
      off.current.az -= (e.clientX - lx) * 0.006;
      off.current.el = Math.min(1.1, Math.max(-0.35, off.current.el + (e.clientY - ly) * 0.004));
      lx = e.clientX; ly = e.clientY;
    };
    const up = () => { drag = false; };
    const wheel = (e) => {
      if (!e.ctrlKey) return;
      e.preventDefault();
      off.current.dist = Math.min(3.0, Math.max(-0.6, off.current.dist + e.deltaY * 0.002));
    };
    el.addEventListener("mousedown", down);
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    el.addEventListener("wheel", wheel, {passive: false});
    return () => {
      el.removeEventListener("mousedown", down);
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      el.removeEventListener("wheel", wheel);
    };
  }, [gl]);
  return off;
}

function CameraFollow({runtime}) {
  const orbit = useCamOrbit();
  const {camera} = useThree();
  const started = useStore((s) => s.started);
  const target = useRef(new THREE.Vector3(0, 0.18, 0));
  const desired = useRef(new THREE.Vector3());
  const here = useRef(new THREE.Vector3());
  const fromPos = useRef(new THREE.Vector3());
  const fromTarget = useRef(new THREE.Vector3());
  const t0 = useRef(0);
  const wasStarted = useRef(false);

  useFrame(() => {
    if (!runtime?.sim) return;
    let p, q;
    try { p = runtime.sim.basePos(); q = runtime.sim.baseQuat(); }
    catch { return; }                       // transient WASM hiccup: skip the frame
    // MJCF (x, y, z) -> three (x, z, -y)
    here.current.set(p[0], p[2] + 0.05, -p[1]);

    if (!started) {
      wasStarted.current = false;
      target.current.lerp(here.current, 0.12);
      const o = orbit.current;
      const r = Math.hypot(CAM_IDLE.x, CAM_IDLE.z) + o.dist;
      const a = Math.atan2(CAM_IDLE.z, CAM_IDLE.x) + o.az;
      desired.current.set(target.current.x + Math.cos(a) * r,
                          target.current.y + CAM_IDLE.y + o.el,
                          target.current.z + Math.sin(a) * r);
      camera.position.lerp(desired.current, 0.06);
      camera.lookAt(target.current);
      return;
    }

    // Start pressed: remember where we were, then ease across to the chase pose.
    if (!wasStarted.current) {
      wasStarted.current = true;
      t0.current = performance.now();
      fromPos.current.copy(camera.position);
      fromTarget.current.copy(target.current);
    }

    // Heading from the base quaternion (MuJoCo order is w, x, y, z).
    const yaw = Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                           1 - 2 * (q[2] * q[2] + q[3] * q[3]));
    // MJCF forward is +x; in three that is (cos yaw, 0, -sin yaw).
    const o = orbit.current;
    const yawO = yaw + o.az;
    const dist = CAM_DIST + o.dist;
    const fx = Math.cos(yawO), fz = -Math.sin(yawO);
    desired.current.set(here.current.x - fx * dist,
                        here.current.y + CAM_HEIGHT + o.el,
                        here.current.z - fz * dist);

    const k = Math.min(1, (performance.now() - t0.current) / CAM_TRANSITION_MS);
    if (k < 1) {                            // eased fly-in, not a snap
      const e = ease(k);
      camera.position.lerpVectors(fromPos.current, desired.current, e);
      target.current.lerpVectors(fromTarget.current, here.current, e);
    } else {                                // steady chase
      target.current.lerp(here.current, 0.18);
      camera.position.lerp(desired.current, 0.10);
    }
    camera.lookAt(target.current);
  });
  return null;
}

function Ground() {
  return (
    <>
      <gridHelper args={[20, 80, 0x2a3038, 0x1b2026]} position={[0, 0.001, 0]} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[20, 20]} />
        <meshStandardMaterial color="#14181d" roughness={0.95} />
      </mesh>
    </>
  );
}

export default function GameCanvas({runtime, onCanvas}) {
  const [dpr, setDpr] = useState(1.5);
  return (
    <Canvas
      shadows
      dpr={dpr}
      camera={{position: [0.9, 0.6, 0.9], fov: 45, near: 0.01, far: 100}}
      onCreated={({gl}) => {
        gl.setClearColor("#0d0f12");
        onCanvas?.(gl.domElement);
        setDpr(Math.min(2, window.devicePixelRatio));
      }}
    >
      <hemisphereLight args={[0xbfd4ff, 0x1a1a1a, 0.55]} />
      <directionalLight
        position={[2, 3, 1.5]} intensity={1.5} castShadow
        shadow-mapSize={[1024, 1024]} shadow-camera-near={0.1} shadow-camera-far={12}
        shadow-camera-left={-2} shadow-camera-right={2}
        shadow-camera-top={2} shadow-camera-bottom={-2}
      />
      <Ground />
      <Rig runtime={runtime} />
      <CameraFollow runtime={runtime} />
    </Canvas>
  );
}
