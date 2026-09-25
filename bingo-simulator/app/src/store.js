// UI <-> game-core state. The store holds a SNAPSHOT the runtime pushes; the runtime
// never reads from it, so React can re-render freely without touching physics.
import {create} from "zustand";

export const useStore = create((set) => ({
  ready: false,
  started: false,            // Start pressed -> third-person chase camera
  error: null,
  snapshot: null,
  hudMode: "public",          // "public" | "engineering"
  sceneMode: "arena",          // "arena" | "home" - cosmetic only, physics ground is unchanged
  paused: false,
  setReady: (v) => set({ready: v}),
  setStarted: (v) => set({started: v}),
  setError: (e) => set({error: e}),
  setSnapshot: (s) => set({snapshot: s}),
  toggleHud: () => set((s) => ({hudMode: s.hudMode === "public" ? "engineering" : "public"})),
  toggleScene: () => set((s) => ({sceneMode: s.sceneMode === "arena" ? "home" : "arena"})),
  setPaused: (v) => set({paused: v}),
}));
