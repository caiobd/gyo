// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";
import { renderInspector, renderMap, residualBand, residualColor } from "../atlas-render.js";

const SVG = "http://www.w3.org/2000/svg";
const hostile = `<img src=x onerror="globalThis.pwned=true">`;
const sample = (idx, label = hostile) => ({ idx, label, path: `${idx}.png` });
const node = {
  prefix: [2], occupancy: 12, purity: 0.75, mean_residual: 1.25,
  residual_norm: .25, parent_distance: 0.5, token_norm: 2, has_children: true,
  samples: { representative: [sample(7)], outliers: [sample(8)] },
};

describe("semantic atlas renderer", () => {
  it("maps normalized residuals to a clamped colorblind-safe sequential scale", () => {
    expect(residualColor(0)).toBe("#315a9b");
    expect(residualColor(1)).toBe("#e5c84b");
    expect(residualColor(-2)).toBe(residualColor(0));
    expect(residualColor(4)).toBe(residualColor(1));
    expect(residualColor(null)).toBeNull();
    expect(residualColor(Number.NaN)).toBeNull();
  });

  it("adds redundant residual bands, patterns, and accessible labels", () => {
    const svg = document.createElementNS(SVG, "svg");
    renderMap(svg, [{ ...node, cx: 100, cy: 90, r: 70 }], { selected: null }, {});
    const territory = svg.querySelector(".territory");
    expect(residualBand(.1)).toBe("low"); expect(residualBand(.5)).toBe("mid"); expect(residualBand(.9)).toBe("high");
    expect(territory.dataset.residualBand).toBe("low");
    expect(territory.getAttribute("aria-label")).toContain("normalized residual 0.250, low");
    expect(territory.querySelector("circle:not(.selection-ring)").style.stroke).toBe(residualColor(.25));
  });

  it("keeps projection guidance in empty and populated inspector states", () => {
    const container = document.createElement("aside");
    renderInspector(container, null, "representative");
    expect(container.querySelector(".projection-help")?.textContent).toContain("among siblings");
    renderInspector(container, node, "representative");
    expect(container.querySelectorAll(".projection-help")).toHaveLength(1);
    expect(container.querySelector(".projection-help")?.textContent).toContain("layout stress includes display fitting");
  });

  it("shows complete analytics, explicit token identity, and parent semantics", () => {
    const container = document.createElement("aside");
    renderInspector(container, node, "parent", { focus: { samples: node.samples } });
    expect(container.querySelector("h2").textContent).toBe("Level 1 · token c2");
    expect(container.textContent).toContain("Mean residual");
    expect(container.textContent).toContain("Normalized residual");
    expect(container.textContent).toContain("Token c2 moves the reconstruction by 0.500 in original Euclidean space");
    expect(container.textContent).toContain("Parent group samples");
    expect(container.textContent).toContain("Child token c2 samples");
  });

  it("omits unavailable metrics and gives root no token identity", () => {
    const container = document.createElement("aside");
    renderInspector(container, { ...node, prefix: [], purity: null, parent_distance: null, token_norm: null }, "representative");
    expect(container.querySelector("h2").textContent).toBe("Root group");
    expect(container.textContent).not.toContain("Purity");
    expect(container.textContent).not.toContain("Token norm");
  });
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

  it("renders hierarchy structure and updates path state for pointer and keyboard focus", () => {
    const svg = document.createElementNS(SVG, "svg"); const path = vi.fn();
    renderMap(svg, [{ ...node, cx: 100, cy: 90, r: 20 }], { selected: null, focus: [4] }, { width: 240, height: 180, path });
    expect(svg.querySelector(".focus-boundary")).not.toBeNull();
    expect(svg.querySelector(".focus-anchor")?.textContent).toContain("Group 4");
    expect(svg.querySelectorAll(".hierarchy-link")).toHaveLength(1);
    expect(svg.querySelector('.children-group[role="group"]')?.getAttribute("aria-label")).toContain("Group 4");
    expect(svg.querySelectorAll('.focus-boundary[aria-hidden="true"], .focus-anchor[aria-hidden="true"], .hierarchy-links[aria-hidden="true"]')).toHaveLength(3);
    expect(svg.querySelector(".focus-boundary")?.dataset.prefix).toBe("4");
    expect(svg.querySelector(".territory")?.getAttribute("aria-level")).toBe("2");
    const territory = svg.querySelector(".territory");
    territory.dispatchEvent(new Event("pointerenter"));
    expect(territory.classList.contains("is-path")).toBe(true);
    expect(svg.querySelector(".focus-anchor").classList.contains("is-path")).toBe(true);
    expect(path).toHaveBeenLastCalledWith(node.prefix);
    territory.dispatchEvent(new FocusEvent("focus")); territory.dispatchEvent(new Event("pointerleave")); expect(path).toHaveBeenLastCalledWith(node.prefix);
    territory.dispatchEvent(new FocusEvent("blur")); expect(path).toHaveBeenLastCalledWith(null);
    renderMap(svg, [], { selected: null, focus: [4] }, { width: 240, height: 180 });
    expect(svg.textContent).toContain("No child groups at this level");
    expect(svg.querySelector(".focus-anchor")).not.toBeNull();
  });

  it("expands aggregate buttons without selecting or entering", () => {
    const svg = document.createElementNS(SVG, "svg"); const handlers = { expand: vi.fn(), select: vi.fn(), enter: vi.fn() };
    renderMap(svg, [{ aggregate: true, label: "+2 groups", count: 2, occupancy: 3, position: [0, 0], cx: 100, cy: 90, r: 20 }], { selected: null, focus: [], aggregateExpanded: true }, handlers);
    const button = svg.querySelector('.aggregate[role="treeitem"]');
    expect(button.getAttribute("aria-expanded")).toBe("false");
    expect(button.getAttribute("aria-label")).toBe("2 more groups, Reveal more groups");
    button.dispatchEvent(new KeyboardEvent("keydown", { key: " ", bubbles: true })); button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(handlers.expand).toHaveBeenCalledTimes(2); expect(handlers.select).not.toHaveBeenCalled(); expect(handlers.enter).not.toHaveBeenCalled();
  });

  it("makes a hard-cap aggregate disabled, untabbable, and inert", () => {
    const svg = document.createElementNS(SVG, "svg"); const expand = vi.fn();
    renderMap(svg, [{ aggregate: true, revealable: false, count: 160, occupancy: 300, position: [0, 0], cx: 100, cy: 90, r: 20 }], { selected: null, focus: [] }, { expand });
    const aggregate = svg.querySelector(".aggregate");
    expect(aggregate.getAttribute("aria-disabled")).toBe("true"); expect(aggregate.getAttribute("tabindex")).toBe("-1");
    aggregate.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    aggregate.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    aggregate.dispatchEvent(new KeyboardEvent("keydown", { key: " ", bubbles: true }));
    expect(expand).not.toHaveBeenCalled();
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
    expect(container.textContent).toContain("Parent group samples");
    expect(container.textContent).toContain("Child token c2 samples");
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

  it("places group entry before the potentially long sample grid", () => {
    const container = document.createElement("aside");
    renderInspector(container, node, "representative", {});
    const enter = container.querySelector(".enter-group");
    const samples = container.querySelector(".sample-grid");
    expect(enter.compareDocumentPosition(samples) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
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
    const image = svg.querySelector("image"); const skeleton = svg.querySelector(".svg-image-skeleton");
    expect(skeleton).not.toBeNull();
    const geometry = [image.getAttribute("x"), image.getAttribute("y"), image.getAttribute("width"), image.getAttribute("height")];
    image.dispatchEvent(new Event("load")); expect(svg.querySelector(".svg-image-skeleton")).toBeNull();
    image.dispatchEvent(new Event("error"));
    const retry = svg.querySelector('[role="button"]');
    expect(retry?.textContent).toContain("Retry");
    expect(image.getAttribute("visibility")).toBe("hidden");
    retry.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(image.getAttribute("visibility")).toBe("visible");
    expect(svg.querySelector(".svg-image-skeleton")).not.toBeNull();
    expect([image.getAttribute("x"), image.getAttribute("y"), image.getAttribute("width"), image.getAttribute("height")]).toEqual(geometry);
    expect(image.getAttribute("href")).toContain("/thumb/7");
  });

  it("preserves inspector slot geometry while loading, on error, and retry", () => {
    const container = document.createElement("aside"); renderInspector(container, node, "representative");
    const slot = container.querySelector(".thumb-slot"), image = slot.querySelector("img");
    expect(slot.querySelector(".skeleton")).not.toBeNull();
    image.dispatchEvent(new Event("load")); expect(slot.querySelector(".skeleton")).toBeNull();
    image.dispatchEvent(new Event("error")); expect(slot.querySelector("button").textContent).toBe("Retry");
    slot.querySelector("button").click(); expect(slot.querySelector(".skeleton")).not.toBeNull();
    expect(slot.querySelector("img")).not.toBeNull();
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
