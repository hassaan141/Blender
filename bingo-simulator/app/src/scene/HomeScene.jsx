// Cosmetic "home" backdrop: a small warm room built from CC0 Poly Haven furniture
// (public/home/models/*) around Bingo's stand spot, toggled against the plain
// engineering Arena in GameCanvas. This never touches physics - the MJCF ground
// plane stays exactly where it is; only what's rendered around it changes.
//
// Assets: polyhaven.com, CC0. Downloaded at 1k resolution to keep the bundle small
// (~11 MB total across 5 props + one floor texture).
import React, {useEffect, useRef} from "react";
import * as THREE from "three";
import {GLTFLoader} from "three/addons/loaders/GLTFLoader.js";
import {useThree} from "@react-three/fiber";

const loader = new GLTFLoader();
const texLoader = new THREE.TextureLoader();

// name -> [x, z, yaw] placement around the origin where Bingo stands, and a uniform
// scale (Poly Haven props are real-world scale in meters, already correct).
const LAYOUT = [
  {model: "Sofa_01", pos: [-1.9, -2.0], yaw: Math.PI * 0.15},
  {model: "CoffeeTable_01", pos: [-1.1, -1.5], yaw: 0.2},
  {model: "Rockingchair_01", pos: [1.7, -1.9], yaw: -Math.PI * 0.35},
  {model: "Shelf_01", pos: [-2.2, 1.8], yaw: -Math.PI / 2},
  {model: "potted_plant_01", pos: [2.1, 1.6], yaw: 0},
];

function loadFloor() {
  const dir = "/home/textures/laminate_floor_02/laminate_floor_02";
  const diff = texLoader.load(`${dir}_diffuse_1k.jpg`);
  const nor = texLoader.load(`${dir}_nor_gl_1k.jpg`);
  const rough = texLoader.load(`${dir}_rough_1k.jpg`);
  for (const t of [diff, nor, rough]) {
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(5, 5);
    t.colorSpace = t === diff ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  }
  return new THREE.MeshStandardMaterial({
    map: diff, normalMap: nor, roughnessMap: rough, roughness: 1,
  });
}

async function loadProp(name) {
  const url = `/home/models/${name}/${name}.gltf`;
  const gltf = await loader.loadAsync(url);
  const obj = gltf.scene;
  obj.traverse((n) => {
    if (n.isMesh) { n.castShadow = true; n.receiveShadow = true; }
  });
  return obj;
}

export default function HomeScene() {
  const {scene} = useThree();
  const group = useRef(null);

  useEffect(() => {
    let alive = true;
    const g = new THREE.Group();
    group.current = g;
    scene.add(g);

    // Warm walls + floor, sized around the ~6m room the props are laid out in.
    const wallMat = new THREE.MeshStandardMaterial({color: "#e8ddc8", roughness: 0.95});
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), loadFloor());
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    g.add(floor);

    const backWall = new THREE.Mesh(new THREE.PlaneGeometry(6, 2.8), wallMat);
    backWall.position.set(0, 1.4, -3);
    backWall.receiveShadow = true;
    g.add(backWall);

    const sideWall = new THREE.Mesh(new THREE.PlaneGeometry(6, 2.8), wallMat);
    sideWall.rotation.y = Math.PI / 2;
    sideWall.position.set(-3, 1.4, 0);
    sideWall.receiveShadow = true;
    g.add(sideWall);

    (async () => {
      for (const item of LAYOUT) {
        try {
          const obj = await loadProp(item.model);
          if (!alive) return;
          obj.position.set(item.pos[0], 0, item.pos[1]);
          obj.rotation.y = item.yaw;
          g.add(obj);
        } catch (e) {
          console.warn(`HomeScene: failed to load ${item.model}`, e);
        }
      }
    })();

    return () => {
      alive = false;
      scene.remove(g);
      g.traverse((n) => {
        if (n.isMesh) {
          n.geometry?.dispose?.();
          const mats = Array.isArray(n.material) ? n.material : [n.material];
          for (const m of mats) m?.dispose?.();
        }
      });
    };
  }, [scene]);

  return null;
}
