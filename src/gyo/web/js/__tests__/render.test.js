import { describe, it, expect } from "vitest";
import { sampleToSlots, tileForBox } from "../render.js";

describe("sampleToSlots", () => {
  it("returns all items when fewer than slots", () => {
    expect(sampleToSlots([1, 2, 3], 10)).toEqual([1, 2, 3]);
  });

  it("returns all items when equal to slots", () => {
    expect(sampleToSlots([1, 2, 3], 3)).toEqual([1, 2, 3]);
  });

  it("evenly sub-samples to exactly slots", () => {
    const out = sampleToSlots([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 5);
    expect(out.length).toBe(5);
    expect(out[0]).toBe(0);
    expect(out[4]).toBe(8);
  });

  it("handles single slot", () => {
    const out = sampleToSlots([10, 20, 30, 40], 1);
    expect(out.length).toBe(1);
    expect(out[0]).toBe(10);
  });

  it("handles empty input", () => {
    expect(sampleToSlots([], 5)).toEqual([]);
  });
});

describe("tileForBox", () => {
  it("returns larger tile for wider nodes", () => {
    const small = tileForBox(80, 60);
    const large = tileForBox(300, 60);
    expect(large).toBeGreaterThan(small);
  });

  it("clamps minimum to 11", () => {
    expect(tileForBox(20, 20)).toBeGreaterThanOrEqual(11);
  });

  it("clamps maximum to 34", () => {
    expect(tileForBox(800, 800)).toBeLessThanOrEqual(34);
  });

  it("returns at least 11 for root-sized box", () => {
    const tile = tileForBox(400, 60);
    expect(tile).toBeGreaterThanOrEqual(11);
    expect(tile).toBeLessThanOrEqual(34);
  });
});
