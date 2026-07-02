import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchAtlas } from "../api.js";

afterEach(() => vi.unstubAllGlobals());

describe("fetchAtlas", () => {
  it("fetches and returns an encoded atlas prefix", async () => {
    const payload = { focus: { prefix: [1, 2] }, children: [] };
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => payload });
    vi.stubGlobal("fetch", fetch);
    await expect(fetchAtlas("1,2")).resolves.toBe(payload);
    expect(fetch).toHaveBeenCalledWith("/api/atlas/1%2C2", undefined);
  });

  it("reports JSON detail and HTTP status on failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: false, status: 404, statusText: "Not Found", json: async () => ({ detail: "prefix not found" }),
    }));
    await expect(fetchAtlas("9")).rejects.toThrow("prefix not found (HTTP 404)");
  });
});
