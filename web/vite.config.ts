import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" so the built static site can be hosted from any subpath (e.g. gh-pages).
export default defineConfig({
  plugins: [react()],
  base: "./",
});
