import { describe, it, expect } from "vitest";
import { buildModel, depthOf, residualColor, prefixKey } from "../model.js";

/* ── fixture: a small tree matching the /api/tree JSON shape ── */
const tree = {
  num_levels: 2,
  dead_codewords: [0, 1],
  nodes: [
    { prefix: [], level: 0, occupancy: 30, mean_residual: 0.4, residual_norm: 0.5, purity: 1 },
    { prefix: [4], level: 1, occupancy: 18, mean_residual: 0.3, residual_norm: 0.2, purity: 1 },
    { prefix: [9], level: 1, occupancy: 12, mean_residual: 0.6, residual_norm: 0.9, purity: 1 },
    { prefix: [4, 1], level: 2, occupancy: 18, mean_residual: 0.3, residual_norm: 0.2, purity: 1 },
  ],
};

/* ── fixture: tree where all children at level 1 are dead ── */
const treeAllDeadChildren = {
  num_levels: 2,
  nodes: [
    { prefix: [], level: 0, occupancy: 0, mean_residual: 0.4, residual_norm: 0.5, purity: null },
    { prefix: [4], level: 1, occupancy: 0, mean_residual: 0.3, residual_norm: 0.2, purity: null, is_dead: true },
    { prefix: [9], level: 1, occupancy: 0, mean_residual: 0.8, residual_norm: 1.0, purity: null, is_dead: true },
  ],
};

/* ── fixture: tree with a single dead node ── */
const treeWithDead = {
  num_levels: 2,
  nodes: [
    { prefix: [], level: 0, occupancy: 30, mean_residual: 0.4, residual_norm: 0.5, purity: 1 },
    { prefix: [4], level: 1, occupancy: 18, mean_residual: 0.3, residual_norm: 0.2, purity: 1 },
    { prefix: [9], level: 1, occupancy: 0, mean_residual: 0.8, residual_norm: 1.0, purity: null, is_dead: true },
    { prefix: [4, 1], level: 2, occupancy: 18, mean_residual: 0.3, residual_norm: 0.2, purity: 1 },
  ],
};

/* ── fixture: tree with label field ── */
const treeWithLabels = {
  num_levels: 2,
  nodes: [
    { prefix: [], level: 0, occupancy: 30, mean_residual: 0.4, residual_norm: 0.5, purity: 1 },
    { prefix: [4], level: 1, occupancy: 18, mean_residual: 0.3, residual_norm: 0.2, purity: 1, label: "cat" },
    { prefix: [9], level: 1, occupancy: 12, mean_residual: 0.6, residual_norm: 0.9, purity: 1, label: "dog" },
    { prefix: [4, 1], level: 2, occupancy: 18, mean_residual: 0.3, residual_norm: 0.2, purity: 1, label: "kitten" },
  ],
};

describe("buildModel", () => {
  it("builds a rooted tree with parent links and depth", () => {
    const root = buildModel(tree);
    expect(root.isRoot).toBe(true);
    expect(root.children.length).toBe(2);
    expect(depthOf(root)).toBe(2);
    const c4 = root.children.find(c => c.code === 4);
    expect(c4.children[0].parent).toBe(c4);
  });

  it("sets root occ from the empty-prefix node", () => {
    const root = buildModel(tree);
    expect(root.occ).toBe(30);
    expect(root.residual_norm).toBe(0.5);
  });

  it("sets code to the last element of prefix", () => {
    const root = buildModel(tree);
    const c4 = root.children.find(c => c.code === 4);
    expect(c4.code).toBe(4);
    const c41 = c4.children[0];
    expect(c41.code).toBe(1);
  });

  it("marks leaf nodes when level === num_levels", () => {
    const root = buildModel(tree);
    const c4 = root.children.find(c => c.code === 4);
    const leaf = c4.children[0];
    expect(leaf.leaf).toBe(true);
    // sibling at level 1 is not a leaf
    expect(c4.leaf).toBe(false);
  });

  it("preserves the dead property from API response", () => {
    const root = buildModel(treeWithDead);
    const deadChild = root.children.find(c => c.code === 9);
    expect(deadChild).toBeDefined();
    // BUG: buildModel never reads is_dead from the API response
    expect(deadChild.dead).toBe(true);
  });

  it("preserves the label property from API response", () => {
    const root = buildModel(treeWithLabels);
    const c4 = root.children.find(c => c.code === 4);
    // BUG: buildModel never reads label from the API response
    expect(c4.label).toBe("cat");
  });
});

describe("depthOf", () => {
  it("returns 0 for a leaf node", () => {
    const root = buildModel(tree);
    const leaf = root.children[0].children[0];
    expect(depthOf(leaf)).toBe(0);
  });

  it("returns 0 when all children are dead", () => {
    const root = buildModel(treeAllDeadChildren);
    // all level-1 children are dead → should be excluded → depth is 0
    expect(depthOf(root)).toBe(0);
  });

  it("returns correct depth for tree with live children only", () => {
    const root = buildModel(tree);
    expect(depthOf(root)).toBe(2);
  });
});

describe("prefixKey", () => {
  it('returns "root" for the root node', () => {
    const root = buildModel(tree);
    expect(prefixKey(root)).toBe("root");
  });

  it("joins prefix with commas for non-root nodes", () => {
    const root = buildModel(tree);
    const c4 = root.children.find(c => c.code === 4);
    expect(prefixKey(c4)).toBe("4");
    const c41 = c4.children[0];
    expect(prefixKey(c41)).toBe("4,1");
  });
});

describe("residualColor", () => {
  it("returns green for t=0", () => {
    const c = residualColor(0);
    expect(c).toBe("rgb(0,212,170)");
  });

  it("returns yellow for t=0.5", () => {
    const c = residualColor(0.5);
    expect(c).toBe("rgb(255,234,167)");
  });

  it("returns red for t=1", () => {
    const c = residualColor(1);
    expect(c).toBe("rgb(255,107,107)");
  });

  it("clamps t below 0 to green", () => {
    expect(residualColor(-0.5)).toBe(residualColor(0));
  });

  it("clamps t above 1 to red", () => {
    expect(residualColor(1.5)).toBe(residualColor(1));
  });

  it("is continuous at t=0.5 boundary", () => {
    const below = residualColor(0.499);
    const at = residualColor(0.5);
    // BUG: rounding discontinuity — 0.499 yields 254,234,167 instead of 255,234,167
    // The interpolation from green→yellow at u=0.998 rounds to 254 instead of 255.
    // Both sides of the boundary should produce identical yellow.
    expect(below).toBe(at);
  });
});
