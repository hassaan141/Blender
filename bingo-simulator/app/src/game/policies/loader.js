// Versioned policy loading. The simulator REFUSES an incompatible policy rather than
// silently running it - the failure mode the brief calls out, and the one Microduck's
// daemon is explicitly built to prevent (publish/manifest.py: the daemon "refuses a
// policy whose manifest disagrees with these, and refuses at load a network whose
// graph does").
//
// Refusing matters more here than it does for a duck. A silently reordered joint
// vector on a quadruped does not throw - it walks wrong, and looks like a bad policy
// rather than a loading bug.

import {ORT_URL, ORT_WASM_DIR, MANIFEST_URL, OBS_SIZE, ACTION_SIZE,
        JOINT_NAMES, CONTROL_HZ, PHYSICS_HZ} from "../constants.js";

let ortPromise = null;

export function loadOrt() {
  if (!ortPromise) {
    ortPromise = import(/* @vite-ignore */ ORT_URL).then((ort) => {
      ort.env.wasm.wasmPaths = ORT_WASM_DIR;
      return ort;
    });
  }
  return ortPromise;
}

export class PolicyRefused extends Error {}

/** Everything the manifest must agree with before a policy is allowed to run. */
export function checkManifest(man) {
  const problems = [];
  const c = man.control || {};
  const o = man.observation || {};
  const a = man.action || {};

  if (o.dim !== OBS_SIZE) {
    problems.push(`observation dim ${o.dim} != ${OBS_SIZE} expected by this runtime`);
  }
  if (a.dim !== ACTION_SIZE) {
    problems.push(`action dim ${a.dim} != ${ACTION_SIZE}`);
  }
  if (c.control_hz !== CONTROL_HZ) {
    problems.push(`control_hz ${c.control_hz} != ${CONTROL_HZ}. This is not a detail: `
      + `a policy trained at another rate sees a different observation sequence.`);
  }
  if (c.physics_hz !== PHYSICS_HZ) {
    problems.push(`physics_hz ${c.physics_hz} != ${PHYSICS_HZ}`);
  }
  const jo = man.joint_order || [];
  if (jo.length !== JOINT_NAMES.length || jo.some((n, i) => n !== JOINT_NAMES[i])) {
    const at = jo.findIndex((n, i) => n !== JOINT_NAMES[i]);
    problems.push(`joint_order differs from the canonical order (first at index ${at}: `
      + `"${jo[at]}" vs "${JOINT_NAMES[at]}")`);
  }
  if (Array.isArray(a.scale) && a.scale.length !== ACTION_SIZE) {
    problems.push(`action.scale has ${a.scale.length} entries, expected ${ACTION_SIZE}`);
  }
  return problems;
}

export async function fetchManifest(url = MANIFEST_URL) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`manifest fetch ${r.status}`);
  return r.json();
}

/**
 * Load a policy named in the manifest. Returns null when the manifest declares no
 * policies - a normal state, not an error, until one is trained.
 */
export async function loadPolicy(man, name = null) {
  const list = man.policies || [];
  if (list.length === 0) return null;

  const entry = name ? list.find((p) => p.name === name) : list[0];
  if (!entry) throw new PolicyRefused(`no policy named "${name}" in the manifest`);

  const problems = checkManifest(man);
  if (problems.length) {
    throw new PolicyRefused(
      `refusing "${entry.name}": the manifest disagrees with this runtime.\n  `
      + problems.join("\n  "));
  }

  const ort = await loadOrt();
  const session = await ort.InferenceSession.create(entry.file, {executionProviders: ["wasm"]});

  // The manifest can still lie. Check the graph itself, as Microduck's daemon does.
  const inName = session.inputNames[0];
  const outName = session.outputNames.includes("actions")
    ? "actions" : session.outputNames[0];

  return {
    name: entry.name,
    session, ort, inName, outName,
    manifest: man,
    actionScale: Float32Array.from(man.action.scale),
    async run(obs) {
      if (obs.length !== OBS_SIZE) {
        throw new PolicyRefused(`built a ${obs.length}-float observation, expected ${OBS_SIZE}`);
      }
      const out = await session.run({[inName]: new ort.Tensor("float32", obs, [1, OBS_SIZE])});
      const act = out[outName].data;
      if (act.length !== ACTION_SIZE) {
        throw new PolicyRefused(
          `policy returned ${act.length} actions, manifest says ${ACTION_SIZE}. `
          + `The network graph and the manifest disagree.`);
      }
      return act;
    },
  };
}
