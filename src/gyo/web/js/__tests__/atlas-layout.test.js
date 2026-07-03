import { describe, expect, it } from "vitest";
import { aggregateDenseChildren, displayStress, fitTerritories } from "../atlas-layout.js";

const node = (position, occupancy) => ({ position, occupancy });

describe("fitTerritories", () => {
  it("aggregates dense children stably with weighted position and occupancy", () => {
    const children = Array.from({ length: 65 }, (_, i) => ({ prefix: [i], occupancy: i + 1, position: [i / 64, -i / 64] }));
    const result = aggregateDenseChildren(children);
    expect(result).toHaveLength(64);
    expect(result.slice(0, 63).map(item => item.prefix[0])).toEqual(Array.from({ length: 63 }, (_, i) => i + 2));
    expect(result.at(-1)).toMatchObject({ aggregate: true, count: 2, occupancy: 3, label: "+2 groups" });
    expect(result.at(-1).position[0]).toBeCloseTo(1 / 64 * 2 / 3);
    expect(aggregateDenseChildren(children)).toEqual(result);
    expect(aggregateDenseChildren(children, 63, true)).toBe(children);
  });

  it("preserves occupancy when aggregating 256 children", () => {
    const children = Array.from({ length: 256 }, (_, i) => ({ prefix: [i], occupancy: (i % 7) + 1, position: [0, 0] }));
    const visible = aggregateDenseChildren(children);
    expect(visible.reduce((sum, item) => sum + item.occupancy, 0)).toBe(children.reduce((sum, item) => sum + item.occupancy, 0));
    expect(visible.at(-1).count).toBe(193);
  });
  it("returns an empty layout for no nodes", () => {
    expect(fitTerritories([], 640, 480)).toEqual([]);
  });

  it("maps normalized positions, occupancy radii, and keeps bounds", () => {
    const nodes = [node([-1, 0], 100), node([1, 0], 25)];
    const placed = fitTerritories(nodes, 800, 400);
    expect(placed[0].cx).toBeLessThan(placed[1].cx);
    expect(placed[0].r).toBeGreaterThan(placed[1].r);
    for (const item of placed) {
      expect(item.cx - item.r).toBeGreaterThanOrEqual(4);
      expect(item.cy - item.r).toBeGreaterThanOrEqual(4);
      expect(item.cx + item.r).toBeLessThanOrEqual(796);
      expect(item.cy + item.r).toBeLessThanOrEqual(396);
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
    expect(performance.now() - started).toBeLessThan(1500);
    expect(fitTerritories(nodes, 800, 600)).toEqual(placed);
    expect(placed.every(item => item.layoutMode === "grid-fallback")).toBe(true);
    expect(placed[0].r).toBeLessThan(placed.at(-1).r);
    expect(placed.every(({ cx, cy, r }) =>
      [cx, cy, r].every(Number.isFinite) && cx >= r && cy >= r && cx + r <= 800 && cy + r <= 600)).toBe(true);
    for (let i = 0; i < placed.length; i++) for (let j = i + 1; j < placed.length; j++) {
      expect(Math.hypot(placed[i].cx - placed[j].cx, placed[i].cy - placed[j].cy))
        .toBeGreaterThanOrEqual(placed[i].r + placed[j].r - 0.01);
    }
  });

  it("scales fallback radii into arbitrarily tiny finite viewports", () => {
    const size = 1e-20;
    const nodes = Array.from({ length: 65 }, (_, index) => node([0, 0], index + 1));
    const placed = fitTerritories(nodes, size, size);
    expect(placed.every(({ cx, cy, r }) =>
      [cx, cy, r].every(Number.isFinite) && r >= 0 && cx >= r && cy >= r && cx + r <= size && cy + r <= size)).toBe(true);
    for (let i = 0; i < placed.length; i++) for (let j = i + 1; j < placed.length; j++) {
      expect(Math.hypot(placed[i].cx - placed[j].cx, placed[i].cy - placed[j].cy) + 1e-35)
        .toBeGreaterThanOrEqual(placed[i].r + placed[j].r);
    }
  });
});

describe("displayStress", () => {
  const placements = [[0, 0], [3, 0], [0, 4]].map(([cx, cy]) => ({ cx, cy }));
  const distances = [[0, 3, 4], [3, 0, 5], [4, 5, 0]];
  it("is zero for exact and globally scaled geometry", () => {
    expect(displayStress(distances, placements)).toBeCloseTo(0, 12);
    expect(displayStress(distances, placements.map(p => ({ cx: p.cx * 7, cy: p.cy * 7 })))).toBeCloseTo(0, 12);
  });
  it("reports distorted geometry and handles degenerate counts", () => {
    expect(displayStress(distances, [{ cx: 0, cy: 0 }, { cx: 1, cy: 0 }, { cx: 2, cy: 0 }])).toBeGreaterThan(.1);
    expect(displayStress([], [])).toBe(0);
    expect(displayStress([[0]], [{ cx: 1, cy: 2 }])).toBe(0);
  });
  it("rejects malformed values", () => {
    expect(() => displayStress([[0, 1]], placements)).toThrow(/square/);
    expect(() => displayStress([[0, NaN], [NaN, 0]], placements.slice(0, 2))).toThrow(/finite/);
    expect(() => displayStress([[1]], [{ cx: 0, cy: 0 }])).toThrow(/diagonal/);
    expect(() => displayStress([[0, 1], [2, 0]], placements.slice(0, 2))).toThrow(/symmetric/);
    expect(() => displayStress([[0, -1], [-1, 0]], placements.slice(0, 2))).toThrow(/non-negative/);
  });
});
