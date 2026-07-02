import { fetchAtlas } from "./api.js";
import { fitTerritories } from "./atlas-layout.js";
import { createState, parentPrefix, prefixKey, selectNode, setSampleMode } from "./atlas-model.js";
import { cancelMapInteractions, renderInspector, renderMap } from "./atlas-render.js";

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

export function createRequestGuard() {
  let id = 0, controller;
  return {
    next() { controller?.abort(); controller = new AbortController(); return { id: ++id, signal: controller.signal }; },
    isCurrent(candidate) { return candidate === id; },
    abort() { controller?.abort(); id += 1; },
  };
}

export function zoomView(view, factor, point, base = view) {
  const minWidth = base.width / 10, maxWidth = base.width * 2;
  const width = clamp(view.width / factor, minWidth, maxWidth);
  const height = width * base.height / base.width;
  const rx = (point.x - view.x) / view.width, ry = (point.y - view.y) / view.height;
  const x = clamp(point.x - rx * width, base.x - width * .5, base.x + base.width - width * .5);
  const y = clamp(point.y - ry * height, base.y - height * .5, base.y + base.height - height * .5);
  return { x, y, width, height };
}

export function startAtlas(doc = document, win = window) {
  const svg = doc.getElementById("atlas");
  const inspector = doc.getElementById("inspector");
  const loading = doc.getElementById("mapLoading");
  const error = doc.getElementById("mapError");
  const status = doc.getElementById("projectionStatus");
  const crumbs = doc.getElementById("breadcrumbs");
  const back = doc.getElementById("backBtn");
  const retry = doc.getElementById("retryBtn") || error.querySelector("button");
  const brand = doc.querySelector(".brand");
  const cache = new Map(), guard = createRequestGuard(), removers = [];
  let state, placements = [], successful = false, currentPrefix = "root";
  let view, baseView, drag = null, suppressClick = false, resizeTimer, destroyed = false;

  const on = (target, type, listener, options) => {
    target.addEventListener(type, listener, options);
    removers.push(() => target.removeEventListener(type, listener, options));
  };
  const selectedNode = () => state?.payload.children.find(node => prefixKey(node.prefix) === prefixKey(state.selected || []));
  const bounds = () => { const rect = svg.getBoundingClientRect(); return { width: Math.max(1, rect.width || svg.clientWidth || 800), height: Math.max(1, rect.height || svg.clientHeight || 600) }; };
  const applyView = () => svg.setAttribute("viewBox", `${view.x} ${view.y} ${view.width} ${view.height}`);
  function resetView() { const size = bounds(); baseView = view = { x: 0, y: 0, ...size }; applyView(); }
  function showStatus() {
    const projection = state.payload.projection || {};
    const fallback = placements.some(item => item.layoutMode === "grid-fallback");
    const parts = [];
    if (Number.isFinite(projection.stress)) parts.push(`Stress ${projection.stress.toFixed(3)}`);
    if (projection.warning) parts.push("projection warning");
    if (fallback) parts.push("grid fallback distorts distances");
    status.textContent = parts.join(" · "); status.classList.toggle("warning", Boolean(projection.warning || fallback));
  }
  function renderBreadcrumbs() {
    crumbs.replaceChildren();
    for (let i = 0; i <= state.focus.length; i++) {
      const prefix = state.focus.slice(0, i), button = doc.createElement("button");
      button.type = "button"; button.textContent = i ? String(state.focus[i - 1]) : "Root";
      if (i === state.focus.length) button.setAttribute("aria-current", "page");
      else button.addEventListener("click", () => load(prefixKey(prefix)));
      crumbs.appendChild(button); if (i < state.focus.length) crumbs.append("›");
    }
    back.disabled = state.focus.length === 0;
  }
  function renderAll(reflow = true) {
    if (!state || destroyed) return;
    if (reflow) { const size = bounds(); placements = fitTerritories(state.payload.children, size.width, size.height); resetView(); }
    renderMap(svg, placements, state, {
      select(node) { state = selectNode(state, node.prefix); renderAll(false); },
      enter(node) { if (node.has_children) load(prefixKey(node.prefix)); },
    });
    renderInspector(inspector, selectedNode(), state.sampleMode, {
      focus: state.payload.focus,
      mode(mode) { state = setSampleMode(state, mode); renderAll(false); },
      enter(node) { if (node.has_children) load(prefixKey(node.prefix)); },
    });
    renderBreadcrumbs(); showStatus();
  }
  async function load(prefix = "root", force = false) {
    if (destroyed) return;
    currentPrefix = prefix; const request = guard.next(); loading.hidden = false; error.hidden = true;
    try {
      const payload = !force && cache.has(prefix) ? cache.get(prefix) : await fetchAtlas(prefix, { signal: request.signal });
      if (!guard.isCurrent(request.id) || destroyed) return;
      cache.set(prefix, payload); state = createState(payload); successful = true; renderAll();
    } catch (reason) {
      if (reason?.name === "AbortError" || !guard.isCurrent(request.id) || destroyed) return;
      error.querySelector("p").textContent = reason instanceof Error ? reason.message : "Unable to load atlas";
      error.hidden = false; if (!successful) svg.replaceChildren();
    } finally { if (guard.isCurrent(request.id) && !destroyed) loading.hidden = true; }
  }
  function endDrag(event, mayClick) {
    if (!drag || event.pointerId !== drag.pointerId) return;
    if (mayClick && drag.moved) suppressClick = true;
    const pointerId = drag.pointerId; drag = null;
    if (typeof svg.hasPointerCapture === "function" && svg.hasPointerCapture(pointerId)) svg.releasePointerCapture?.(pointerId);
  }

  on(retry, "click", () => load(currentPrefix, true));
  on(brand, "click", event => { event.preventDefault(); load("root"); });
  on(back, "click", () => state && load(prefixKey(parentPrefix(state.focus))));
  on(doc.getElementById("resetViewBtn"), "click", resetView);
  on(win, "keydown", event => { if (event.key === "Escape" && state?.focus.length) load(prefixKey(parentPrefix(state.focus))); });
  on(win, "resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => renderAll(true), 100); });
  on(svg, "click", event => { if (suppressClick) { suppressClick = false; event.stopImmediatePropagation(); event.preventDefault(); } }, true);
  on(svg, "wheel", event => {
    if (!view) return; event.preventDefault(); const rect = svg.getBoundingClientRect();
    const point = { x: view.x + (event.clientX - rect.left) / rect.width * view.width, y: view.y + (event.clientY - rect.top) / rect.height * view.height };
    view = zoomView(view, event.deltaY < 0 ? 1.15 : 1 / 1.15, point, baseView); applyView();
  }, { passive: false });
  on(svg, "pointerdown", event => { if (event.button != null && event.button !== 0) return; drag = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, view: { ...view }, moved: false }; svg.setPointerCapture?.(event.pointerId); });
  on(svg, "pointermove", event => {
    if (!drag || event.pointerId !== drag.pointerId || !view) return;
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y; if (Math.hypot(dx, dy) < 5 && !drag.moved) return; drag.moved = true;
    const rect = svg.getBoundingClientRect(); view = { ...view, x: drag.view.x - dx / rect.width * drag.view.width, y: drag.view.y - dy / rect.height * drag.view.height }; applyView();
  });
  on(svg, "pointerup", event => endDrag(event, true));
  on(svg, "pointercancel", event => endDrag(event, false));
  on(svg, "lostpointercapture", event => endDrag(event, false));

  function destroy() { if (destroyed) return; destroyed = true; guard.abort(); clearTimeout(resizeTimer); cancelMapInteractions(svg); removers.splice(0).forEach(remove => remove()); }
  load(); return { load, render: renderAll, destroy };
}

if (typeof document !== "undefined" && document.getElementById("atlas")) startAtlas();
