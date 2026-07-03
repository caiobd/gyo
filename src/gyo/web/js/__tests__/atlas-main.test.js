// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { createRequestGuard, MAX_RENDERED_TERRITORIES, startAtlas, zoomView } from "../main.js";

const payload = (prefix = [], children = []) => ({
  focus: { prefix, occupancy: 10, samples: { representative: [], outliers: [] } },
  children,
  dataset_id: "dataset-a",
  projection: { raw_stress: .125, stress: .125 },
});
const child = { prefix: [1], occupancy: 4, position: [0, 0], has_children: true, samples: { representative: [], outliers: [] } };

function shell() {
  document.body.replaceChildren();
  document.body.insertAdjacentHTML("afterbegin", `<a class="brand" href="#">gyo</a><nav id="breadcrumbs"></nav><output id="projectionStatus"></output><button id="backBtn"></button><button id="resetViewBtn"></button><label>Level <select id="levelControl"></select></label><button id="collapseDenseBtn" hidden></button><div id="mapLoading" hidden></div><svg id="atlas"></svg><div id="mapError" hidden><p></p><button id="retryBtn"></button></div><aside id="inspector"></aside>`);
  const svg = document.getElementById("atlas");
  svg.getBoundingClientRect = () => ({ width: 800, height: 600, left: 0, top: 0 });
  svg.setPointerCapture = vi.fn(); svg.releasePointerCapture = vi.fn();
  return svg;
}
const flush = () => new Promise(resolve => setTimeout(resolve, 0));

afterEach(() => { vi.restoreAllMocks(); document.body.replaceChildren(); });

describe("atlas orchestration", () => {
  it("invalidates stale responses and aborts the previous request", () => {
    const guard = createRequestGuard(); const first = guard.next(); const second = guard.next();
    expect(first.signal.aborted).toBe(true); expect(guard.isCurrent(first.id)).toBe(false); expect(guard.isCurrent(second.id)).toBe(true);
  });

  it("loads root from the brand and reuses cached payloads", async () => {
    shell();
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(async url => ({ ok: true, json: async () => url.endsWith("root") ? payload([], [child]) : payload([1]) }));
    const app = startAtlas(); await flush();
    await app.load("1"); await flush();
    document.querySelector(".brand").dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true })); await flush();
    expect(fetch).toHaveBeenCalledTimes(5);
    expect(document.getElementById("breadcrumbs").textContent).toContain("Root");
    app.destroy();
  });

  it("revalidates dataset identity before using a cached prefix", async () => {
    shell(); let dataset = "dataset-a";
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(async url => {
      if (url === "/api/dataset") return { ok: true, json: async () => ({ dataset_id: dataset }) };
      return { ok: true, json: async () => ({ ...payload([], [child]), dataset_id: dataset }) };
    });
    const app = startAtlas(); await flush();
    dataset = "dataset-b";
    await app.load("root"); await flush();
    expect(fetch.mock.calls.filter(([url]) => url === "/api/atlas/root")).toHaveLength(2);
    app.destroy();
  });

  it("retry bypasses cache and preserves the last map on an error", async () => {
    const svg = shell(); let fail = false;
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(async () => fail ? ({ ok: false, status: 503, statusText: "Down", json: async () => ({ detail: "offline" }) }) : ({ ok: true, json: async () => payload([], [child]) }));
    const app = startAtlas(); await flush(); const territory = svg.querySelector(".territory");
    fail = true; await app.load("2", true); await flush();
    expect(svg.contains(territory)).toBe(true);
    document.getElementById("retryBtn").click(); await flush();
    expect(fetch).toHaveBeenCalledTimes(4);
    app.destroy();
  });

  it("discloses projection/grid warnings and resets after resize", async () => {
    const svg = shell();
    const many = Array.from({ length: 65 }, (_, i) => ({ ...child, prefix: [i], position: [0, 0] }));
    vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true, json: async () => payload([], many) });
    vi.useFakeTimers(); const app = startAtlas(); await vi.runAllTimersAsync();
    expect(document.getElementById("projectionStatus").textContent).toMatch(/Raw MDS stress 0.125.*layout stress unavailable while \d+ groups aggregated/);
    expect(document.getElementById("projectionStatus").classList.contains("warning")).toBe(false);
    svg.setAttribute("viewBox", "1 2 3 4"); window.dispatchEvent(new Event("resize")); await vi.runAllTimersAsync();
    expect(svg.getAttribute("viewBox")).toBe("0 0 800 600");
    app.destroy(); vi.useRealTimers();
  });

  it("expands and collapses dense groups without loading the aggregate", async () => {
    const svg = shell(); const many = Array.from({ length: 65 }, (_, i) => ({ ...child, prefix: [i], occupancy: i + 1, position: [0, 0] }));
    const fetch = vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true, json: async () => ({ ...payload([], many), num_levels: 3 }) });
    vi.useFakeTimers(); const app = startAtlas(); await vi.runAllTimersAsync();
    expect(svg.querySelectorAll(".territory").length).toBeLessThan(65);
    const firstCount = svg.querySelectorAll(".territory").length;
    svg.querySelector(".territory.aggregate").dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    expect(svg.querySelectorAll(".territory").length).toBeGreaterThan(firstCount); expect(fetch).toHaveBeenCalledTimes(2);
    document.getElementById("collapseDenseBtn").click(); expect(svg.querySelectorAll(".territory")).toHaveLength(firstCount);
    app.destroy(); vi.useRealTimers();
  });

  it("uses viewport capacity and progressively reveals another page on zoom", async () => {
    const svg = shell(); svg.getBoundingClientRect = () => ({ width: 360, height: 240, left: 0, top: 0 });
    const many = Array.from({ length: 40 }, (_, i) => ({ ...child, prefix: [i], occupancy: i + 1, position: [0, 0] }));
    vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true, json: async () => ({ ...payload([], many), num_levels: 2 }) });
    vi.useFakeTimers(); const app = startAtlas(); await vi.runAllTimersAsync();
    const initial = svg.querySelectorAll(".territory").length; expect(initial).toBeLessThan(40);
    const wheel = () => svg.dispatchEvent(new WheelEvent("wheel", { deltaY: -1, clientX: 180, clientY: 120, bubbles: true, cancelable: true }));
    wheel(); wheel();
    expect(svg.querySelectorAll(".territory").length).toBeGreaterThan(initial);
    expect(document.getElementById("projectionStatus").textContent).toMatch(/groups aggregated/);
    app.destroy(); vi.useRealTimers();
  });

  it("preserves the zoomed camera and hard-caps repeated dense reveals", async () => {
    const svg = shell(); const many = Array.from({ length: 256 }, (_, i) => ({ ...child, prefix: [i], occupancy: i + 1, position: [0, 0] }));
    vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true, json: async () => ({ ...payload([], many), num_levels: 2 }) });
    vi.useFakeTimers(); const app = startAtlas(); await vi.runAllTimersAsync();
    const initial = svg.querySelectorAll(".territory:not(.aggregate)").length;
    const wheel = () => svg.dispatchEvent(new WheelEvent("wheel", { deltaY: -1, clientX: 400, clientY: 300, bubbles: true, cancelable: true }));
    wheel(); wheel(); const zoomed = svg.getAttribute("viewBox");
    expect(svg.querySelectorAll(".territory:not(.aggregate)").length).toBeGreaterThan(initial);
    expect(svg.getAttribute("viewBox")).toBe(zoomed);
    const aggregate = () => svg.querySelector(".territory.aggregate");
    for (let i = 0; i < 10; i++) aggregate()?.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    expect(svg.querySelectorAll(".territory:not(.aggregate)").length).toBeLessThanOrEqual(MAX_RENDERED_TERRITORIES);
    expect(aggregate()).not.toBeNull(); expect(aggregate().getAttribute("aria-expanded")).toBe("false");
    expect(aggregate().getAttribute("aria-label")).toContain("enter a branch or collapse");
    expect(svg.getAttribute("viewBox")).toBe(zoomed);
    app.destroy(); vi.useRealTimers();
  });

  it("loads the selected reachable comparison level at a leaf", async () => {
    shell(); const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(async url => ({ ok: true, json: async () => ({ ...payload(url.endsWith("1.2") ? [1, 2] : [1], []), num_levels: 4 }) }));
    const app = startAtlas(); await flush(); await app.load("1.2"); await flush();
    const control = document.getElementById("levelControl");
    expect([...control.options].map(option => option.value)).toEqual(["1", "2", "3", "4"]);
    expect([...control.options].map(option => option.disabled)).toEqual([false, false, false, true]);
    expect(control.options[3].textContent).toContain("select a group at level 3 first");
    control.value = "2"; control.dispatchEvent(new Event("change", { bubbles: true })); await flush();
    expect(fetch.mock.calls.some(([url]) => url === "/api/atlas/1")).toBe(true); expect(document.getElementById("backBtn").disabled).toBe(false);
    app.destroy();
  });

  it("clamps zoom and suppresses only the click synthesized by a drag", async () => {
    const svg = shell(); vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true, json: async () => payload([], [child]) });
    vi.useFakeTimers(); const app = startAtlas(); await vi.runAllTimersAsync();
    const event = (type, values) => { const e = new Event(type, { bubbles: true }); Object.assign(e, values); return e; };
    svg.dispatchEvent(event("pointerdown", { pointerId: 3, clientX: 10, clientY: 10 }));
    svg.dispatchEvent(event("pointermove", { pointerId: 3, clientX: 30, clientY: 30 }));
    svg.dispatchEvent(event("pointerup", { pointerId: 3, clientX: 30, clientY: 30 }));
    svg.querySelector(".territory").dispatchEvent(new MouseEvent("click", { bubbles: true })); await vi.runAllTimersAsync();
    expect(document.getElementById("inspector").textContent).toContain("Select a territory");
    svg.querySelector(".territory").dispatchEvent(new MouseEvent("click", { bubbles: true })); await vi.runAllTimersAsync();
    expect(document.getElementById("inspector").textContent).toContain("Group 1");
    let view = { x: 0, y: 0, width: 800, height: 600 };
    for (let i = 0; i < 100; i++) view = zoomView(view, 1.15, { x: 400, y: 300 }, { width: 800, height: 600 });
    expect(view.width).toBeGreaterThanOrEqual(80);
    app.destroy(); vi.useRealTimers();
  });

  it("does not capture a tap before it becomes a drag", async () => {
    const svg = shell();
    vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true, json: async () => payload([], [child]) });
    vi.useFakeTimers(); const app = startAtlas(); await vi.runAllTimersAsync();
    const pointer = (type, values) => { const event = new Event(type, { bubbles: true }); Object.assign(event, values); return event; };
    svg.dispatchEvent(pointer("pointerdown", { pointerId: 8, clientX: 10, clientY: 10, button: 0 }));
    expect(svg.setPointerCapture).not.toHaveBeenCalled();
    svg.dispatchEvent(pointer("pointermove", { pointerId: 8, clientX: 30, clientY: 30 }));
    expect(svg.setPointerCapture).toHaveBeenCalledWith(8);
    app.destroy(); vi.useRealTimers();
  });

  it("destroy aborts work and removes interaction listeners", async () => {
    const svg = shell(); let signal;
    vi.spyOn(globalThis, "fetch").mockImplementation((_url, options) => { signal = options.signal; return new Promise(() => {}); });
    const app = startAtlas(); app.destroy();
    expect(signal.aborted).toBe(true);
    svg.dispatchEvent(new Event("wheel", { cancelable: true }));
    expect(svg.getAttribute("viewBox")).toBeNull();
  });

  it.each(["pointercancel", "lostpointercapture"])("does not suppress a genuine click after %s", async eventType => {
    const svg = shell();
    svg.hasPointerCapture = vi.fn(() => false);
    vi.spyOn(globalThis, "fetch").mockResolvedValue({ ok: true, json: async () => payload([], [child]) });
    vi.useFakeTimers(); const app = startAtlas(); await vi.runAllTimersAsync();
    const pointer = (type, values) => { const event = new Event(type, { bubbles: true }); Object.assign(event, values); return event; };
    svg.dispatchEvent(pointer("pointerdown", { pointerId: 5, clientX: 10, clientY: 10 }));
    svg.dispatchEvent(pointer("pointermove", { pointerId: 5, clientX: 30, clientY: 30 }));
    expect(() => svg.dispatchEvent(pointer(eventType, { pointerId: 5 }))).not.toThrow();
    svg.querySelector(".territory").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await vi.runAllTimersAsync();
    expect(document.getElementById("inspector").textContent).toContain("Group 1");
    expect(svg.releasePointerCapture).not.toHaveBeenCalled();
    app.destroy(); vi.useRealTimers();
  });
});
