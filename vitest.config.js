import { defineConfig } from "vitest/config";
export default defineConfig({
  test: { include: ["src/gyo/web/js/__tests__/**/*.test.js"], environment: "node" },
});
