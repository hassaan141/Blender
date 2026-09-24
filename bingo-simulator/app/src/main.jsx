import React from "react";
import {createRoot} from "react-dom/client";
import App from "./App.jsx";
import "./index.css";

// NOTE: no <React.StrictMode>. It double-invokes the boot effect, which initialises
// MuJoCo WASM twice; the vendored build throws an uncaught Embind error on
// MjModel::actuator_actearly (a bool memory_view it cannot convert), and the second
// init re-enters React's scheduler ("Should not already be working"), killing the
// root so the HUD never mounts.
createRoot(document.getElementById("root")).render(<App />);
