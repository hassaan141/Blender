// The visual robot: one Three.js Group per body, built from kinematics.json.
//
// HARD RULE (brief §5, §16): this rig is driven ENTIRELY from MuJoCo state. It has no
// animation of its own, no interpolation and no idle motion. If physics is paused,
// the robot is still. Everything a viewer sees on screen came out of the solver.

import * as THREE from "three";
import {STLLoader} from "three/addons/loaders/STLLoader.js";
import {mergeVertices, toCreasedNormals} from "three/addons/utils/BufferGeometryUtils.js";
import {KINEMATICS, MODEL_DIR} from "../game/constants.js";

export async function loadKinematics() {
  const r = await fetch(KINEMATICS);
  if (!r.ok) throw new Error(`kinematics fetch ${r.status}`);
  return r.json();
}

const CREASE_ANGLE = Math.PI / 5;   // 36 deg

const MAT = new THREE.MeshStandardMaterial({
  color: 0xb9c0c7, metalness: 0.25, roughness: 0.55,
});
const MAT_ACCENT = new THREE.MeshStandardMaterial({
  color: 0xd98b5f, metalness: 0.2, roughness: 0.5,
});

/** Build the rig. Returns {root, bodies, setFromPhysics}. */
export async function buildRenderRig(kin) {
  const loader = new STLLoader();

  // placer holds world pose; root fixes MJCF +Z up -> three.js +Y up.
  const placer = new THREE.Group();
  placer.name = "bingo_placer";
  const root = new THREE.Group();
  root.name = "bingo_root";
  root.rotation.x = -Math.PI / 2;
  placer.add(root);

  const bodies = new Map();
  const byName = new Map(kin.bodies.map((b) => [b.name, b]));

  // parents first
  const ordered = [];
  const seen = new Set();
  const visit = (b) => {
    if (seen.has(b.name)) return;
    if (b.parent && byName.has(b.parent)) visit(byName.get(b.parent));
    seen.add(b.name);
    ordered.push(b);
  };
  kin.bodies.forEach(visit);

  const geoms = await Promise.all(
    ordered.map((b) =>
      b.mesh
        ? loader.loadAsync(`${MODEL_DIR}/${kin.mesh_dir}/${b.mesh}`).catch(() => null)
        : Promise.resolve(null)
    )
  );

  ordered.forEach((b, i) => {
    const g = new THREE.Group();
    g.name = b.name;
    g.position.set(...(b.pos || [0, 0, 0]));
    if (b.rpy && b.rpy.some((v) => v)) {
      g.rotation.set(b.rpy[0], b.rpy[1], b.rpy[2], "XYZ");
    }
    // rest orientation, so joint rotation composes on top of it
    g.userData.restQuat = g.quaternion.clone();
    g.userData.axis = b.joint_axis ? new THREE.Vector3(...b.joint_axis).normalize() : null;
    g.userData.joint = b.joint;

    const geo = geoms[i];
    if (geo) {
      // STL is non-indexed: every triangle carries its own three vertices, so
      // computeVertexNormals() produces per-FACE normals and the whole robot renders
      // flat-shaded - most visibly on the head, which is one large smooth volume.
      // Weld first, then rebuild normals with a crease angle so genuinely sharp
      // edges stay sharp and curved surfaces read as curved. (Microduck rebuilds
      // creased normals for the same reason: "STL facet normals are omitted on
      // purpose".) toCreasedNormals hashes on a 0.01-unit grid and these meshes are
      // in metres, so scale to millimetres for the hash and back again.
      const welded = mergeVertices(geo, 1e-5);
      welded.deleteAttribute("normal");
      const mm = welded.clone();
      mm.scale(1000, 1000, 1000);
      const shaded = toCreasedNormals(mm, CREASE_ANGLE);
      shaded.scale(1e-3, 1e-3, 1e-3);
      geo.dispose();
      welded.dispose();
      mm.dispose();
      const mesh = new THREE.Mesh(
        shaded, b.name === "head_roll" || b.name.includes("ear") ? MAT_ACCENT : MAT);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      const mp = b.mesh_pos || [0, 0, 0];
      mesh.position.set(mp[0], mp[1], mp[2]);
      const mr = b.mesh_rpy || [0, 0, 0];
      if (mr.some((v) => v)) mesh.rotation.set(mr[0], mr[1], mr[2], "XYZ");
      g.add(mesh);
    }

    const parent = b.parent && bodies.has(b.parent) ? bodies.get(b.parent) : root;
    parent.add(g);
    bodies.set(b.name, g);
  });

  const _q = new THREE.Quaternion();

  /**
   * Drive the rig from a physics snapshot.
   * @param basePos [x,y,z] MJCF world
   * @param baseQuat [w,x,y,z]
   * @param jointPos {name: radians}
   */
  function setFromPhysics(basePos, baseQuat, jointPos) {
    placer.position.set(basePos[0], basePos[2], -basePos[1]);   // MJCF -> three
    // MJCF quat (w,x,y,z) -> three (x,y,z,w), with the same axis convention swap
    placer.quaternion.set(baseQuat[1], baseQuat[3], -baseQuat[2], baseQuat[0]);

    for (const [name, g] of bodies) {
      const jn = g.userData.joint;
      if (!jn || !g.userData.axis) continue;
      const a = jointPos[jn];
      if (a === undefined) continue;
      _q.setFromAxisAngle(g.userData.axis, a);
      g.quaternion.copy(g.userData.restQuat).multiply(_q);
    }
  }

  return {placer, root, bodies, setFromPhysics};
}
