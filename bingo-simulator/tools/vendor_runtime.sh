#!/bin/bash
# Copy the MuJoCo and ONNX Runtime web builds out of node_modules into public/vendor/.
#
# They are NOT committed (see app/.gitignore): they are ~40 MB of binaries that
# package.json already pins, so `npm install && tools/vendor_runtime.sh` reproduces
# them exactly. Run this once after installing, and again after bumping either
# version.
#
# Why vendored at all, when Microduck loads them from jsdelivr: the app then has no
# third-party network dependency, which matters for an offline demo and on locked-down
# networks (the environment this was built in blocks jsdelivr outright). Set
# VITE_USE_CDN=1 to use the CDN instead.
set -e
cd "$(dirname "$0")/../app"
mkdir -p public/vendor/mujoco public/vendor/ort
cp node_modules/@mujoco/mujoco/mujoco.js  public/vendor/mujoco/
cp node_modules/@mujoco/mujoco/mujoco.wasm public/vendor/mujoco/
cp node_modules/onnxruntime-web/dist/ort.min.mjs public/vendor/ort/
# only the plain wasm backend: jsep (WebGPU), jspi and asyncify builds are ~60 MB
# and unused here.
cp node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs  public/vendor/ort/
cp node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm public/vendor/ort/
echo "vendored:"; du -sh public/vendor/mujoco public/vendor/ort
