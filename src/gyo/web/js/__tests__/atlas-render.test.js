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
    expect(territory?.dataset.prefix).toBe("2");
    expect(svg.querySelector("img")).toBeNull();
    territory.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    territory.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    territory.dispatchEvent(new KeyboardEvent("keydown", { key: " ", bubbles: true }));
    expect(handlers.select).toHaveBeenCalledWith(expect.objectContaining({ prefix: [2] }));
    expect(handlers.enter).toHaveBeenCalledWith(expect.objectContaining({ prefix: [2] }));
  });

  it("sets a supplied initial SVG viewport", () => {
    const svg = document.createElementNS(SVG, "svg");
    renderMap(svg, [], { selected: null }, { width: 720, height: 480 });
    expect(svg.getAttribute("viewBox")).toBe("0 0 720 480");
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

  it("keeps an SVG preview slot and offers a retry after image failure", () => {
    const svg = document.createElementNS(SVG, "svg");
    renderMap(svg, [{ ...node, cx: 100, cy: 90, r: 70 }], { selected: null }, {});
    const image = svg.querySelector("image");
    image.dispatchEvent(new Event("error"));
    const retry = svg.querySelector('[role="button"]');
    expect(retry?.textContent).toContain("Retry");
    expect(image.getAttribute("visibility")).toBe("hidden");
    retry.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(image.getAttribute("visibility")).toBe("visible");
    expect(image.getAttribute("href")).toContain("/thumb/7");
  });

  it("uses roving tabindex and arrow, Home, and End navigation", () => {
    const svg = document.createElementNS(SVG, "svg");
    document.body.appendChild(svg);
    const nodes = [0, 1, 2].map((value, index) => ({ ...node, prefix: [value], cx: 60 + index * 100, cy: 90, r: 40 }));
    renderMap(svg, nodes, { selected: [1] }, {});
    const territories = [...svg.querySelectorAll('[role="treeitem"]')];
    expect(territories.map(item => item.tabIndex)).toEqual([-1, 0, -1]);
    territories[1].focus();
    territories[1].dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
    expect(document.activeElement).toBe(territories[2]);
    territories[2].dispatchEvent(new KeyboardEvent("keydown", { key: "Home", bubbles: true }));
    expect(document.activeElement).toBe(territories[0]);
    territories[0].dispatchEvent(new KeyboardEvent("keydown", { key: "End", bubbles: true }));
    expect(document.activeElement).toBe(territories[2]);
    svg.remove();
  });

  it("cancels pending selection when the map rerenders", () => {
    vi.useFakeTimers();
    const svg = document.createElementNS(SVG, "svg");
    const select = vi.fn();
    renderMap(svg, [{ ...node, cx: 100, cy: 90, r: 70 }], { selected: null }, { select });
    svg.querySelector('[role="treeitem"]').dispatchEvent(new MouseEvent("click", { bubbles: true }));
    renderMap(svg, [], { selected: null }, { select });
    vi.runAllTimers();
    expect(select).not.toHaveBeenCalled();
    vi.useRealTimers();
  });

  it("lets a leaf double click resolve to one selection", () => {
    vi.useFakeTimers();
    const svg = document.createElementNS(SVG, "svg");
    const handlers = { select: vi.fn(), enter: vi.fn() };
    renderMap(svg, [{ ...node, has_children: false, cx: 100, cy: 90, r: 70 }], { selected: null }, handlers);
    const territory = svg.querySelector('[role="treeitem"]');
    territory.dispatchEvent(new MouseEvent("click", { bubbles: true })); territory.dispatchEvent(new MouseEvent("click", { bubbles: true })); territory.dispatchEvent(new MouseEvent("dblclick", { bubbles: true }));
    vi.runAllTimers();
    expect(handlers.select).toHaveBeenCalledOnce();
    expect(handlers.enter).not.toHaveBeenCalled();
    vi.useRealTimers();
  });

  it("uses ordinary pressed mode buttons", () => {
    const container = document.createElement("aside");
    renderInspector(container, node, "outliers", {});
    const buttons = [...container.querySelectorAll(".tabs button")];
    expect(buttons.map(button => button.getAttribute("aria-pressed"))).toEqual(["false", "true", "false"]);
    expect(container.querySelector('[role="tab"]')).toBeNull();
  });
});
