import { describe, it, expect } from "vitest";
import { computeLayout, winTopFromScrollPx, scrollBounds, snapTarget, clamp, SPINE_W, DEAD_W, BRANCH_COLORS } from "../layout.js";

const leaf = (code, occ = 10) => ({ code, occ, residual: 0.5, residual_norm: 0.5, purity: 1, leaf: true, children: [] });
const deadLeaf = (code) => ({ code, occ: 0, dead: true, residual: 0, residual_norm: 0, purity: null, leaf: true, children: [] });
const node = (code, kids) => ({ code, children: kids, occ: kids.reduce((s, c) => s + c.occ, 0), residual: 0.5, residual_norm: 0.5, purity: 1, leaf: false });
const focusWith = (kids) => { const r = node(null, kids); r.isRoot = true; return r; };

describe("computeLayout", () => {
  it("gives expanded siblings equal width", () => {
    const f = focusWith([leaf(4), leaf(10), leaf(15)]);
    const p = computeLayout(f, 1000, { collapsedSet: new Set() });
    const lvl1 = p.filter(x => x.depth === 1).map(x => Math.round(x.w));
    expect(new Set(lvl1).size).toBe(1);
  });

  it("orders collapsed slivers to the left and shrinks them", () => {
    const a = leaf(4), b = leaf(10), c = leaf(15);
    const f = focusWith([a, b, c]);
    const p = computeLayout(f, 1000, { collapsedSet: new Set([c]) });
    const lvl1 = p.filter(x => x.depth === 1).sort((m, n) => m.x - n.x);
    expect(lvl1[0].node).toBe(c);
    expect(lvl1[0].spine).toBe(true);
    expect(lvl1[0].w).toBeLessThan(lvl1[1].w);
  });

  it("assigns one branch hue per focus child, inherited by descendants", () => {
    const a = node(4, [leaf(1), leaf(2)]);
    const f = focusWith([a, leaf(10)]);
    const p = computeLayout(f, 1000, { collapsedSet: new Set() });
    const childHues = p.filter(x => x.node === a.children[0] || x.node === a.children[1]).map(x => x.branch);
    expect(new Set(childHues).size).toBe(1);
    expect(BRANCH_COLORS).toContain(childHues[0]);
  });

  it("gives dead nodes DEAD_W width", () => {
    const f = focusWith([leaf(4), deadLeaf(9)]);
    const p = computeLayout(f, 1000, { collapsedSet: new Set() });
    const dead = p.find(x => x.node.code === 9);
    expect(dead.w).toBe(DEAD_W);
  });

  it("marks dead nodes as dead in the placement", () => {
    const f = focusWith([leaf(4), deadLeaf(9)]);
    const p = computeLayout(f, 1000, { collapsedSet: new Set() });
    const dead = p.find(x => x.node.code === 9);
    expect(dead.node.dead).toBe(true);
  });

  it("gives collapsed non-dead nodes SPINE_W width", () => {
    const a = leaf(4);
    const f = focusWith([a, leaf(10)]);
    const p = computeLayout(f, 1000, { collapsedSet: new Set([a]) });
    const spine = p.find(x => x.node === a);
    expect(spine.w).toBe(SPINE_W);
    expect(spine.spine).toBe(true);
  });

  it("does not mark focus node as spine even if collapsed", () => {
    const f = focusWith([leaf(4)]);
    const p = computeLayout(f, 1000, { collapsedSet: new Set([f]) });
    const root = p.find(x => x.node === f);
    expect(root.spine).toBe(false);
  });

  it("handles empty children array (leaf focus)", () => {
    const f = focusWith([]);
    const p = computeLayout(f, 1000, { collapsedSet: new Set() });
    expect(p.length).toBe(1);
    expect(p[0].node).toBe(f);
  });

  it("handles single child", () => {
    const f = focusWith([leaf(4)]);
    const p = computeLayout(f, 1000, { collapsedSet: new Set() });
    const child = p.find(x => x.depth === 1);
    expect(child.w).toBe(1000);
  });

  it("handles narrow viewport", () => {
    const f = focusWith([leaf(4), leaf(10), leaf(15)]);
    const p = computeLayout(f, 200, { collapsedSet: new Set() });
    const lvl1 = p.filter(x => x.depth === 1);
    expect(lvl1.length).toBe(3);
    expect(lvl1.every(x => x.w > 0)).toBe(true);
  });

  it("handles all children collapsed", () => {
    const a = leaf(4), b = leaf(10);
    const f = focusWith([a, b]);
    const p = computeLayout(f, 1000, { collapsedSet: new Set([a, b]) });
    const lvl1 = p.filter(x => x.depth === 1);
    expect(lvl1.every(x => x.spine)).toBe(true);
  });

  it("depth field is correct for nested nodes", () => {
    const deep = node(4, [node(1, [leaf(7)])]);
    const f = focusWith([deep]);
    const p = computeLayout(f, 1000, { collapsedSet: new Set() });
    const deepLeaf = p.find(x => x.node.code === 7);
    expect(deepLeaf.depth).toBe(3);
  });

  it("branch color propagates to grandchildren", () => {
    const a = node(4, [node(1, [leaf(7)])]);
    const b = leaf(10);
    const f = focusWith([a, b]);
    const p = computeLayout(f, 1000, { collapsedSet: new Set() });
    const grandchild = p.find(x => x.node.code === 7);
    const child = p.find(x => x.node.code === 1);
    expect(grandchild.branch).toBe(child.branch);
    expect(BRANCH_COLORS).toContain(grandchild.branch);
  });

  it("collapsed node's children are not rendered", () => {
    const a = node(4, [leaf(1), leaf(2)]);
    const f = focusWith([a, leaf(10)]);
    const p = computeLayout(f, 1000, { collapsedSet: new Set([a]) });
    const childrenOfA = p.filter(x => x.depth === 2);
    expect(childrenOfA.length).toBe(0);
  });

  it("collapsed node removes gap from layout, redistributing width", () => {
    const a = leaf(4), b = leaf(10), c = leaf(15);
    const f = focusWith([a, b, c]);
    // no collapsed
    const p1 = computeLayout(f, 1000, { collapsedSet: new Set() });
    // collapse c — now only a and b are expanded, get more width
    const p2 = computeLayout(f, 1000, { collapsedSet: new Set([c]) });
    const aW1 = p1.find(x => x.node === a).w;
    const aW2 = p2.find(x => x.node === a).w;
    expect(aW2).toBeGreaterThan(aW1);
  });
});

describe("scroll math", () => {
  it("derives winTop from a scroll offset and clamps", () => {
    const rowH = 100, PEEK = 24;
    expect(winTopFromScrollPx(PEEK, rowH, PEEK, 3)).toBe(0);
    expect(winTopFromScrollPx(PEEK - 2 * rowH, rowH, PEEK, 3)).toBe(2);
    expect(winTopFromScrollPx(PEEK - 99 * rowH, rowH, PEEK, 3)).toBe(3);
  });

  it("scrollBounds spans PEEK down to PEEK-maxWinTop*rowH", () => {
    expect(scrollBounds(3, 100, 24)).toEqual({ min: 24 - 300, max: 24 });
  });

  it("scrollBounds with maxWinTop=0", () => {
    const b = scrollBounds(0, 100, 24);
    expect(b.min).toBe(24);
    expect(b.max).toBe(24);
  });

  it("snapTarget snaps to nearest level", () => {
    const rowH = 100, PEEK = 24;
    expect(snapTarget(PEEK - 130, rowH, PEEK, 3)).toEqual({ winTop: 1, scrollPx: 24 - 100 });
    expect(snapTarget(PEEK, rowH, PEEK, 3)).toEqual({ winTop: 0, scrollPx: PEEK });
    expect(snapTarget(PEEK - 250, rowH, PEEK, 3)).toEqual({ winTop: 3, scrollPx: 24 - 300 });
  });

  it("snapTarget at exact level boundary", () => {
    const rowH = 100, PEEK = 24;
    expect(snapTarget(PEEK - 100, rowH, PEEK, 3)).toEqual({ winTop: 1, scrollPx: PEEK - 100 });
  });

  it("clamp limits value between bounds", () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(-1, 0, 10)).toBe(0);
    expect(clamp(15, 0, 10)).toBe(10);
    expect(clamp(5, 5, 5)).toBe(5);
  });
});
