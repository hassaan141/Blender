// UI <-> game-core state. The store holds a SNAPSHOT the runtime pushes; the runtime
// never reads from it, so React can re-render freely without touching physics.
import {create} from "zustand";

export const useStore = create((set) => ({
  ready: false,
  error: null,
  snapshot: null,
  hudMode: "public",          // "public" | "engineering"
  paused: false,
  setReady: (v) => set({ready: v}),
  setError: (e) => set({error: e}),
  setSnapshot: (s) => set({snapshot: s}),
  toggleHud: () => set((s) => ({hudMode: s.hudMode === "public" ? "engineering" : "public"})),
  setPaused: (v) => set({paused: v}),
}));
