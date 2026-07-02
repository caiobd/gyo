import { describe, expect, it } from "vitest";
import { createState, enterNode, parentPrefix, prefixKey, selectNode, setSampleMode } from "../atlas-model.js";

describe("atlas model", () => {
  it("keys validated prefixes", () => {
    expect(prefixKey([])).toBe("root");
    expect(prefixKey([2, 7])).toBe("2,7");
    for (const value of [null, {}, [1, -1], [1, 1.5], ["1"]]) {
      expect(() => prefixKey(value)).toThrow();
    }
  });

  it("creates initial state without aliasing focus", () => {
    const payload = {
      focus: { prefix: [2], level: 1 },
      children: [{ prefix: [2, 7], occupancy: 3 }],
      projection: { positions: [[0, 0]] },
    };
    const state = createState(payload);
    expect(state).toEqual({ payload, focus: [2], selected: null, sampleMode: "representative" });
    expect(state.focus).not.toBe(payload.focus.prefix);
    expect(state.payload).toBe(payload);
  });

  it("validates the atlas payload focus shape", () => {
    for (const payload of [null, {}, { focus: null }, { focus: [] }, { focus: {} }]) {
      expect(() => createState(payload)).toThrow();
    }
  });

  it("integrates API state with selection and navigation", () => {
    const payload = {
      focus: { prefix: [3], level: 1 },
      children: [{ prefix: [3, 8], occupancy: 4 }],
      projection: { positions: [[0.25, -0.5]] },
    };
    const selected = selectNode(createState(payload), payload.children[0].prefix);
    expect(enterNode(selected).focus).toEqual([3, 8]);
    expect(payload.focus.prefix).toEqual([3]);
    expect(payload.children[0].prefix).toEqual([3, 8]);
  });

  it("selects and enters nodes immutably", () => {
    const payload = { focus: { prefix: [1], level: 1 }, children: [], projection: {} };
    const initial = createState(payload);
    const prefix = [1, 4];
    const outlierState = setSampleMode(initial, "outliers");
    const selected = selectNode(outlierState, prefix);
    prefix.push(9);
    expect(selected.focus).toEqual([1]);
    expect(selected.selected).toEqual([1, 4]);
    expect(selected.sampleMode).toBe("representative");
    expect(initial).toEqual({ payload, focus: [1], selected: null, sampleMode: "representative" });

    const entered = enterNode(selected);
    expect(entered.focus).toEqual([1, 4]);
    expect(entered.focus).not.toBe(selected.selected);
    expect(entered.selected).toBeNull();
    expect(enterNode(initial)).toBe(initial);
  });

  it("sets only supported sample modes", () => {
    const state = createState({ focus: { prefix: [], level: 0 }, children: [], projection: {} });
    for (const mode of ["representative", "outliers", "parent"]) {
      expect(setSampleMode(state, mode).sampleMode).toBe(mode);
    }
    expect(() => setSampleMode(state, "random")).toThrow();
  });

  it("returns a copied parent prefix and validates input", () => {
    const prefix = [2, 7];
    const parent = parentPrefix(prefix);
    expect(parent).toEqual([2]);
    expect(parent).not.toBe(prefix);
    expect(parentPrefix([])).toEqual([]);
    expect(() => parentPrefix([-1])).toThrow();
  });
});
