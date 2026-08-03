import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, the FastAPI backend runs on :8000 and we proxy /geojson + /api/*
// to it so the SPA can talk to a real backend without CORS gymnastics. In
// production, FastAPI serves the built SPA from the same origin (see api.py
// StaticFiles mount), so no proxy is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/geojson": "http://localhost:8000",
      "/api": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/metrics": "http://localhost:8000",
      "/metadata": "http://localhost:8000",
      "/predictions": "http://localhost:8000",
      "/predict": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    target: "es2020",
    rollupOptions: {
      output: {
        // The dashboard route needs Leaflet immediately, while charting pages
        // are lazy-loaded. Keep stable framework/map code in cacheable vendor
        // chunks so the application entry stays below Vite's 500 kB warning
        // and route edits do not invalidate the largest dependencies.
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          if (
            id.includes("/react/") ||
            id.includes("/react-dom/") ||
            id.includes("/scheduler/")
          ) {
            return "vendor-react";
          }
          if (
            id.includes("/leaflet/") ||
            id.includes("/leaflet.markercluster/")
          ) {
            return "vendor-map";
          }
          return undefined;
        },
      },
    },
  },
});
