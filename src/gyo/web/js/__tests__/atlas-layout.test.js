import { describe, expect, it } from "vitest";
import { fitTerritories } from "../atlas-layout.js";

const node = (position, occupancy) => ({ position, occupancy });

describe("fitTerritories", () => {
  it("returns an empty layout for no nodes", () => {
    expect(fitTerritories([], 640, 480)).toEqual([]);
  });

  it("maps normalized positions, occupancy radii, and keeps bounds", () => {
    const nodes = [node([-1, 0], 100), node([1, 0], 25)];
    const placed = fitTerritories(nodes, 800, 400);
    expect(placed[0].cx).toBeLessThan(placed[1].cx);
    expect(placed[0].r).toBeGreaterThan(placed[1].r);
    for (const item of placed) {
      expect(item.cx - item.r).toBeGreaterThanOrEqual(0);
      expect(item.cy - item.r).toBeGreaterThanOrEqual(0);
      expect(item.cx + item.r).toBeLessThanOrEqual(800);
      expect(item.cy + item.r).toBeLessThanOrEqual(400);
    }
    expect(nodes).toEqual([node([-1, 0], 100), node([1, 0], 25)]);
  });

  it("validates viewport, nodes, positions, and occupancy", () => {
    expect(() => fitTerritories([], 0, 10)).toThrow();
    expect(() => fitTerritories([], 10, NaN)).toThrow();
    expect(() => fitTerritories({}, 10, 10)).toThrow();
    expect(() => fitTerritories([node([0], 1)], 10, 10)).toThrow();
    expect(() => fitTerritories([node([2, 0], 1)], 10, 10)).toThrow();
    expect(() => fitTerritories([node([0, 0], -1)], 10, 10)).toThrow();
  });

  it("scales coordinates and radii when resized", () => {
    const nodes = [node([0, 0], 9)];
    const small = fitTerritories(nodes, 200, 100)[0];
    const large = fitTerritories(nodes, 400, 200)[0];
    expect(large.cx).toBeCloseTo(small.cx * 2);
    expect(large.cy).toBeCloseTo(small.cy * 2);
    expect(large.r).toBeCloseTo(small.r * 2);
  });

  it("separates coincident territories and is deterministic", () => {
    const nodes = [node([0, 0], 10), node([0, 0], 10), node([0, 0], 10)];
    const first = fitTerritories(nodes, 500, 500);
    expect(fitTerritories(nodes, 500, 500)).toEqual(first);
    for (let i = 0; i < first.length; i++) for (let j = i + 1; j < first.length; j++) {
      const distance = Math.hypot(first[i].cx - first[j].cx, first[i].cy - first[j].cy);
      expect(distance).toBeGreaterThanOrEqual(first[i].r + first[j].r - 0.01);
    }
  });

  it("shrinks impossible coincident groups to finite in-bounds circles", () => {
    const nodes = Array.from({ length: 12 }, () => node([0, 0], 1));
    const placed = fitTerritories(nodes, 20, 20);
    expect(placed.every(({ cx, cy, r }) => [cx, cy, r].every(Number.isFinite))).toBe(true);
    expect(placed.every(({ cx, cy, r }) => cx >= r && cy >= r && cx + r <= 20 && cy + r <= 20)).toBe(true);
    for (let i = 0; i < placed.length; i++) for (let j = i + 1; j < placed.length; j++) {
      expect(Math.hypot(placed[i].cx - placed[j].cx, placed[i].cy - placed[j].cy))
        .toBeGreaterThanOrEqual(placed[i].r + placed[j].r - 0.01);
    }
  });

  it("packs 200 coincident nodes quickly, safely, and deterministically", () => {
    const nodes = Array.from({ length: 200 }, (_, index) => ({ ...node([0, 0], index + 1), index }));
    const started = performance.now();
    const placed = fitTerritories(nodes, 800, 600);
    expect(performance.now() - started).toBeLessThan(500);
    expect(fitTerritories(nodes, 800, 600)).toEqual(placed);
    expect(placed.every(item => item.layoutMode === "grid-fallback")).toBe(true);
    expect(placed.every(({ cx, cy, r }) =>
      [cx, cy, r].every(Number.isFinite) && cx >= r && cy >= r && cx + r <= 800 && cy + r <= 600)).toBe(true);
    for (let i = 0; i < placed.length; i++) for (let j = i + 1; j < placed.length; j++) {
      expect(Math.hypot(placed[i].cx - placed[j].cx, placed[i].cy - placed[j].cy))
        .toBeGreaterThanOrEqual(placed[i].r + placed[j].r - 0.01);
    }
  });
});
