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
    const snap = runtime.snapshot();
    rig.setFromPhysics(snap.basePos, runtime.sim.baseQuat(), snap.jointPos);
    // The HUD does not need 120 updates a second.
    acc.current += dt;
    if (acc.current > 0.1) { acc.current = 0; setSnapshot(snap); }
  });
  return null;
}

function CameraFollow({runtime}) {
  const {camera} = useThree();
  const target = useRef(new THREE.Vector3(0, 0.18, 0));
  useFrame(() => {
    if (!runtime?.sim) return;
    const p = runtime.sim.basePos();
    // MJCF (x, y, z) -> three (x, z, -y)
    target.current.lerp(new THREE.Vector3(p[0], p[2] + 0.05, -p[1]), 0.12);
    const desired = new THREE.Vector3(
      target.current.x + 0.85, target.current.y + 0.45, target.current.z + 0.85);
    camera.position.lerp(desired, 0.06);
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
