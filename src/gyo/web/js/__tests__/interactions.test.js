import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { makeClickHandlers, showTip, hideTip } from "../interactions.js";

/* ── click handler tests ── */
describe("makeClickHandlers", () => {
  it("fires collapse on single click after the delay", () => {
    vi.useFakeTimers();
    const onCollapse = vi.fn(), onDrill = vi.fn();
    const h = makeClickHandlers({ onCollapse, onDrill, delay: 220 });
    h.onClick({ stopPropagation() {} }, "N");
    vi.advanceTimersByTime(230);
    expect(onCollapse).toHaveBeenCalledWith("N");
    expect(onDrill).not.toHaveBeenCalled();
    vi.useRealTimers();
  });

  it("double click cancels collapse and drills", () => {
    vi.useFakeTimers();
    const onCollapse = vi.fn(), onDrill = vi.fn();
    const h = makeClickHandlers({ onCollapse, onDrill, delay: 220 });
    h.onClick({ stopPropagation() {} }, "N");
    h.onDblClick({ stopPropagation() {} }, "N");
    vi.advanceTimersByTime(230);
    expect(onDrill).toHaveBeenCalledWith("N");
    expect(onCollapse).not.toHaveBeenCalled();
    vi.useRealTimers();
  });

  it("does nothing if onClick is called without a follow-up", () => {
    vi.useFakeTimers();
    const onCollapse = vi.fn(), onDrill = vi.fn();
    const h = makeClickHandlers({ onCollapse, onDrill, delay: 220 });
    h.onClick({ stopPropagation() {} }, "N");
    vi.advanceTimersByTime(100);
    expect(onCollapse).not.toHaveBeenCalled();
    expect(onDrill).not.toHaveBeenCalled();
    vi.useRealTimers();
  });

  it("calls stopPropagation on click", () => {
    vi.useFakeTimers();
    const spy = vi.fn();
    const h = makeClickHandlers({ onCollapse() {}, onDrill() {} });
    h.onClick({ stopPropagation: spy }, "N");
    expect(spy).toHaveBeenCalled();
    vi.useRealTimers();
  });

  it("calls stopPropagation on dblclick", () => {
    const spy = vi.fn();
    const h = makeClickHandlers({ onCollapse() {}, onDrill() {} });
    h.onDblClick({ stopPropagation: spy }, "N");
    expect(spy).toHaveBeenCalled();
  });

  it("uses custom delay", () => {
    vi.useFakeTimers();
    const onCollapse = vi.fn();
    const h = makeClickHandlers({ onCollapse, onDrill() {}, delay: 500 });
    h.onClick({ stopPropagation() {} }, "N");
    vi.advanceTimersByTime(400);
    expect(onCollapse).not.toHaveBeenCalled();
    vi.advanceTimersByTime(110);
    expect(onCollapse).toHaveBeenCalledWith("N");
    vi.useRealTimers();
  });

  it("second click before delay resets the timer", () => {
    vi.useFakeTimers();
    const onCollapse = vi.fn();
    const h = makeClickHandlers({ onCollapse, onDrill() {}, delay: 220 });
    h.onClick({ stopPropagation() {} }, "A");
    vi.advanceTimersByTime(100);
    h.onClick({ stopPropagation() {} }, "B");
    vi.advanceTimersByTime(230);
    // only the second click should fire
    expect(onCollapse).toHaveBeenCalledTimes(1);
    expect(onCollapse).toHaveBeenCalledWith("B");
    vi.useRealTimers();
  });
});

/* ── showTip tests ── */
function makeTipEl() {
  return { style: { display: "none", left: "", top: "" }, innerHTML: "" };
}

const residualColor = (t) => `rgb(${t})`;

describe("showTip", () => {
  let tipEl;
  const origInnerWidth = globalThis.innerWidth;
  const origInnerHeight = globalThis.innerHeight;
  beforeEach(() => {
    tipEl = makeTipEl();
    globalThis.innerWidth = 1920;
    globalThis.innerHeight = 1080;
  });
  afterEach(() => {
    globalThis.innerWidth = origInnerWidth;
    globalThis.innerHeight = origInnerHeight;
  });

  it("shows the tip element", () => {
    showTip({ clientX: 100, clientY: 100 }, { code: 4, occ: 10, residual: 0.5, residual_norm: 0.5, purity: 1, dead: false }, tipEl, { residualColor });
    expect(tipEl.style.display).toBe("block");
  });

  it("displays node code as prefix", () => {
    showTip({ clientX: 100, clientY: 100 }, { code: 4, prefix: [4], occ: 10, residual: 0.5, residual_norm: 0.5, purity: 1, dead: false }, tipEl, { residualColor });
    expect(tipEl.innerHTML).toContain("c4");
  });

  it("displays root prefix", () => {
    showTip({ clientX: 100, clientY: 100 }, { isRoot: true, occ: 30, residual: 0.4, residual_norm: 0.5, purity: 1, dead: false }, tipEl, { residualColor });
    expect(tipEl.innerHTML).toContain("root");
  });

  it("uses norm function when provided", () => {
    const norm = vi.fn((r) => r * 2);
    showTip({ clientX: 100, clientY: 100 }, { code: 4, occ: 10, residual: 0.5, residual_norm: 0.3, purity: 1, dead: false }, tipEl, { residualColor, norm });
    // BUG: norm function is accepted but never called — residualColor receives residual_norm instead
    expect(norm).toHaveBeenCalledWith(0.5);
  });

  it("falls back to residual_norm when no norm provided", () => {
    showTip({ clientX: 100, clientY: 100 }, { code: 4, occ: 10, residual: 0.5, residual_norm: 0.7, purity: 1, dead: false }, tipEl, { residualColor });
    expect(tipEl.innerHTML).toContain("0.500");
  });

  it("shows dead node status", () => {
    showTip({ clientX: 100, clientY: 100 }, { code: 4, occ: 0, dead: true }, tipEl, { residualColor });
    expect(tipEl.innerHTML).toContain("dead codeword");
  });

  it("shows label for leaf nodes", () => {
    showTip({ clientX: 100, clientY: 100 }, { code: 4, occ: 10, residual: 0.5, residual_norm: 0.5, purity: 1, dead: false, leaf: true, label: "cat" }, tipEl, { residualColor });
    expect(tipEl.innerHTML).toContain("cat");
  });

  it("does not show label for non-leaf nodes", () => {
    showTip({ clientX: 100, clientY: 100 }, { code: 4, occ: 10, residual: 0.5, residual_norm: 0.5, purity: 1, dead: false, leaf: false }, tipEl, { residualColor });
    expect(tipEl.innerHTML).not.toContain("label");
  });

  it("positions tip near cursor", () => {
    showTip({ clientX: 200, clientY: 300 }, { code: 4, occ: 10, residual: 0.5, residual_norm: 0.5, purity: 1, dead: false }, tipEl, { residualColor });
    expect(parseInt(tipEl.style.left)).toBeGreaterThan(200);
    expect(parseInt(tipEl.style.top)).toBeGreaterThan(300);
  });

  it("flips tip when near right edge", () => {
    showTip({ clientX: 1900, clientY: 100 }, { code: 4, occ: 10, residual: 0.5, residual_norm: 0.5, purity: 1, dead: false }, tipEl, { residualColor });
    // should flip to left side of cursor
    expect(parseInt(tipEl.style.left)).toBeLessThan(1900);
  });

  it("flips tip when near bottom edge", () => {
    // clientY=1000 → y=1014, 1014+120=1134 > 1080 → should flip
    showTip({ clientX: 100, clientY: 1000 }, { code: 4, occ: 10, residual: 0.5, residual_norm: 0.5, purity: 1, dead: false }, tipEl, { residualColor });
    expect(parseInt(tipEl.style.top)).toBeLessThan(1000);
  });

  it("displays occupancy", () => {
    showTip({ clientX: 100, clientY: 100 }, { code: 4, occ: 42, residual: 0.5, residual_norm: 0.5, purity: 1, dead: false }, tipEl, { residualColor });
    expect(tipEl.innerHTML).toContain("42");
  });

  it("displays purity as percentage", () => {
    showTip({ clientX: 100, clientY: 100 }, { code: 4, occ: 10, residual: 0.5, residual_norm: 0.5, purity: 0.85, dead: false }, tipEl, { residualColor });
    expect(tipEl.innerHTML).toContain("85%");
  });
});

/* ── hideTip tests ── */
describe("hideTip", () => {
  it("hides the tip element", () => {
    const tipEl = { style: { display: "block" } };
    hideTip(tipEl);
    expect(tipEl.style.display).toBe("none");
  });
});
