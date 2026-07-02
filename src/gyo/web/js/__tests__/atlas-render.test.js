// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";
import { renderInspector, renderMap } from "../atlas-render.js";

const SVG = "http://www.w3.org/2000/svg";
const hostile = `<img src=x onerror="globalThis.pwned=true">`;
const sample = (idx, label = hostile) => ({ idx, label, path: `${idx}.png` });
const node = {
  prefix: [2], occupancy: 12, purity: 0.75, mean_residual: 1.25,
  parent_distance: 0.5, token_norm: 2, has_children: true,
  samples: { representative: [sample(7)], outliers: [sample(8)] },
};

describe("semantic atlas renderer", () => {
  it("creates accessible territories safely and dispatches pointer and keyboard actions", () => {
    const svg = document.createElementNS(SVG, "svg");
    const handlers = { select: vi.fn(), enter: vi.fn() };
    renderMap(svg, [{ ...node, cx: 100, cy: 90, r: 70 }], { selected: [2] }, handlers);
    const territory = svg.querySelector('[role="treeitem"]');
    expect(territory?.getAttribute("tabindex")).toBe("0");
    expect(territory?.getAttribute("aria-selected")).toBe("true");
    expect(svg.querySelector("img")).toBeNull();
    territory.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    territory.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    territory.dispatchEvent(new KeyboardEvent("keydown", { key: " ", bubbles: true }));
    expect(handlers.select).toHaveBeenCalledWith(expect.objectContaining({ prefix: [2] }));
    expect(handlers.enter).toHaveBeenCalledWith(expect.objectContaining({ prefix: [2] }));
  });

  it("renders inspector text without injection and omits unavailable metrics", () => {
    const container = document.createElement("aside");
    renderInspector(container, { ...node, purity: null, mean_residual: undefined }, "representative", {});
    expect(container.textContent).toContain(hostile);
    expect(container.querySelector("img")?.src).toContain("/thumb/7");
    expect(container.querySelector("[onerror]")).toBeNull();
    expect(container.textContent).not.toContain("Purity");
    expect(container.textContent).not.toContain("Residual0");
  });

  it("juxtaposes focus and selected representatives in parent mode", () => {
    const container = document.createElement("aside");
    const focus = { ...node, prefix: [1], samples: { representative: [sample(1, "focus sample")], outliers: [] } };
    renderInspector(container, node, "parent", { focus });
    expect(container.textContent).toContain("Current focus");
    expect(container.textContent).toContain("Selected group");
    expect(container.textContent).toContain("focus sample");
    expect(container.textContent).toContain(hostile);
  });

  it("supports mode and explicit group entry actions", () => {
    const container = document.createElement("aside");
    const handlers = { mode: vi.fn(), enter: vi.fn() };
    renderInspector(container, node, "representative", handlers);
    [...container.querySelectorAll("button")].find(button => button.textContent === "Outliers").click();
    [...container.querySelectorAll("button")].find(button => button.textContent === "Enter group").click();
    expect(handlers.mode).toHaveBeenCalledWith("outliers");
    expect(handlers.enter).toHaveBeenCalledWith(node);
  });

  it("does not leave a selection behind after double click entry", () => {
    vi.useFakeTimers();
    const svg = document.createElementNS(SVG, "svg");
    const handlers = { select: vi.fn(), enter: vi.fn() };
    renderMap(svg, [{ ...node, cx: 100, cy: 90, r: 70 }], { selected: null }, handlers);
    const territory = svg.querySelector('[role="treeitem"]');
    territory.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    territory.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    territory.dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
    vi.runAllTimers();
    expect(handlers.enter).toHaveBeenCalledOnce();
    expect(handlers.select).not.toHaveBeenCalled();
    vi.useRealTimers();
  });
});
