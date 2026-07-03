import { fetchAtlas, fetchDatasetId } from "./api.js";
import { aggregateDenseChildren, displayStress, fitTerritories, viewportCapacity } from "./atlas-layout.js";
import { createState, parentPrefix, prefixKey, selectNode, setSampleMode } from "./atlas-model.js";
import { cancelMapInteractions, renderInspector, renderMap } from "./atlas-render.js";

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
export const MAX_RENDERED_TERRITORIES = 96;

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
  const levelControl = doc.getElementById("levelControl");
  const collapseDense = doc.getElementById("collapseDenseBtn");
  const retry = doc.getElementById("retryBtn") || error.querySelector("button");
  const brand = doc.querySelector(".brand");
  const cache = new Map(), guard = createRequestGuard(), removers = [];
  let state, placements = [], successful = false, currentPrefix = "root";
  let datasetId = null, layoutStress = 0, requestedRevealScale = 1, aggregated = false, hiddenCount = 0, pageCapacity = 63;
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
    if (Number.isFinite(projection.raw_stress ?? projection.stress)) parts.push(`Raw MDS stress ${(projection.raw_stress ?? projection.stress).toFixed(3)}`);
    if (aggregated) parts.push(`layout stress unavailable while ${hiddenCount} groups aggregated`);
    else if (Number.isFinite(layoutStress)) parts.push(`Layout stress ${layoutStress.toFixed(3)}`);
    parts.push("projected approximation among siblings");
    if (fallback) parts.push("grid fallback: semantic distances distorted");
    status.textContent = parts.join(" · "); status.classList.toggle("warning", Boolean(!aggregated && (layoutStress > .10 || fallback)));
  }
  function renderBreadcrumbs() {
    crumbs.replaceChildren();
    for (let i = 0; i <= state.focus.length; i++) {
      const prefix = state.focus.slice(0, i), button = doc.createElement("button");
      button.type = "button"; button.textContent = i ? String(state.focus[i - 1]) : "Root";
      button.dataset.depth = String(i);
      if (i === state.focus.length) button.setAttribute("aria-current", "page");
      else button.addEventListener("click", () => load(prefixKey(prefix)));
      crumbs.appendChild(button); if (i < state.focus.length) crumbs.append("›");
    }
    back.disabled = state.focus.length === 0;
  }
  function renderLevelControl() {
    if (!levelControl) return;
    const current = state.focus.length + 1, total = Math.max(current, Number(state.payload.num_levels) || current);
    levelControl.replaceChildren();
    for (let level = 1; level <= total; level++) {
      const option = doc.createElement("option"); option.value = String(level); option.disabled = level > current;
      option.textContent = level > current ? `Level ${level} — select a group at level ${level - 1} first` : `Level ${level}`;
      option.selected = level === current; levelControl.appendChild(option);
    }
    levelControl.disabled = false;
  }
  function renderAll(reflow = true, preserveView = false) {
    if (!state || destroyed) return;
    if (reflow) {
      const size = bounds(); pageCapacity = viewportCapacity(size.width, size.height);
      const zoomScale = baseView && view ? baseView.width / view.width : 1;
      const revealScale = Math.max(requestedRevealScale, clamp(zoomScale, 1, 2));
      const visibleLimit = Math.min(state.payload.children.length, Math.floor(pageCapacity * revealScale), MAX_RENDERED_TERRITORIES);
      const children = aggregateDenseChildren(state.payload.children, visibleLimit, false);
      const aggregate = children.find(item => item.aggregate);
      if (aggregate) aggregate.revealable = visibleLimit < Math.min(state.payload.children.length, MAX_RENDERED_TERRITORIES);
      aggregated = children.some(item => item.aggregate); hiddenCount = children.find(item => item.aggregate)?.count || 0; placements = fitTerritories(children, size.width, size.height);
      const matrix = state.payload.projection?.distances;
      layoutStress = aggregated ? null : Array.isArray(matrix) && matrix.length === placements.length
        ? displayStress(matrix, placements)
        : (state.payload.projection?.raw_stress ?? state.payload.projection?.stress ?? 0);
      if (!preserveView) resetView();
    }
    const size = bounds();
    renderMap(svg, placements, state, {
      width: size.width, height: size.height,
      select(node) { state = selectNode(state, node.prefix); renderAll(false); },
      enter(node) { if (node.has_children) load(prefixKey(node.prefix)); },
      expand() { requestedRevealScale = Math.min(2, requestedRevealScale + .5); renderAll(true, true); },
      path(prefix) { crumbs.querySelectorAll("button").forEach(button => button.classList.toggle("is-path", Boolean(prefix) && Number(button.dataset.depth) <= state.focus.length)); },
    });
    if (preserveView) applyView();
    renderInspector(inspector, selectedNode(), state.sampleMode, {
      focus: state.payload.focus,
      mode(mode) { state = setSampleMode(state, mode); renderAll(false); },
      enter(node) { if (node.has_children) load(prefixKey(node.prefix)); },
    });
    renderBreadcrumbs(); renderLevelControl();
    if (collapseDense) {
      const expanded = placements.filter(item => !item.aggregate).length > Math.min(pageCapacity, state.payload.children.length);
      collapseDense.hidden = !expanded; collapseDense.setAttribute("aria-pressed", String(expanded));
      collapseDense.setAttribute("aria-label", expanded ? "Collapse expanded small groups" : "Small groups collapsed");
    }
    showStatus();
  }
  async function load(prefix = "root", force = false) {
    if (destroyed) return;
    currentPrefix = prefix; const request = guard.next(); loading.hidden = false; error.hidden = true;
    try {
      const serverDatasetId = await fetchDatasetId({ signal: request.signal });
      if (datasetId && serverDatasetId !== datasetId) cache.clear();
      datasetId = serverDatasetId;
      const cacheKey = datasetId ? `${datasetId}:${prefix}` : null;
      const payload = !force && cacheKey && cache.has(cacheKey) ? cache.get(cacheKey) : await fetchAtlas(prefix, { signal: request.signal });
      if (!guard.isCurrent(request.id) || destroyed) return;
      if (payload.dataset_id && datasetId && payload.dataset_id !== datasetId) cache.clear();
      datasetId = payload.dataset_id || datasetId;
      cache.set(datasetId ? `${datasetId}:${prefix}` : prefix, payload); state = createState(payload); requestedRevealScale = 1; successful = true; renderAll();
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
  if (levelControl) on(levelControl, "change", () => state && load(prefixKey(state.focus.slice(0, Number(levelControl.value) - 1))));
  if (collapseDense) on(collapseDense, "click", () => { requestedRevealScale = 1; renderAll(true, true); });
  on(doc.getElementById("resetViewBtn"), "click", () => { resetView(); renderAll(true, true); });
  on(win, "keydown", event => { if (event.key === "Escape" && state?.focus.length) load(prefixKey(parentPrefix(state.focus))); });
  on(win, "resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => renderAll(true), 100); });
  on(svg, "click", event => { if (suppressClick) { suppressClick = false; event.stopImmediatePropagation(); event.preventDefault(); } }, true);
  on(svg, "wheel", event => {
    if (!view) return; event.preventDefault(); const rect = svg.getBoundingClientRect();
    const point = { x: view.x + (event.clientX - rect.left) / rect.width * view.width, y: view.y + (event.clientY - rect.top) / rect.height * view.height };
    const zoomingIn = event.deltaY < 0; view = zoomView(view, zoomingIn ? 1.15 : 1 / 1.15, point, baseView); applyView();
    const zoomCapacity = Math.min(state?.payload.children.length || 0, Math.floor(pageCapacity * clamp(baseView.width / view.width, 1, 2)), MAX_RENDERED_TERRITORIES);
    if (zoomingIn && aggregated && zoomCapacity > placements.filter(item => !item.aggregate).length) renderAll(true, true);
  }, { passive: false });
  on(svg, "pointerdown", event => { if (event.button != null && event.button !== 0) return; drag = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, view: { ...view }, moved: false }; });
  on(svg, "pointermove", event => {
    if (!drag || event.pointerId !== drag.pointerId || !view) return;
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y; if (Math.hypot(dx, dy) < 5 && !drag.moved) return;
    if (!drag.moved) svg.setPointerCapture?.(event.pointerId);
    drag.moved = true;
    const rect = svg.getBoundingClientRect(); view = { ...view, x: drag.view.x - dx / rect.width * drag.view.width, y: drag.view.y - dy / rect.height * drag.view.height }; applyView();
  });
  on(svg, "pointerup", event => endDrag(event, true));
  on(svg, "pointercancel", event => endDrag(event, false));
  on(svg, "lostpointercapture", event => endDrag(event, false));

  function destroy() { if (destroyed) return; destroyed = true; guard.abort(); clearTimeout(resizeTimer); cancelMapInteractions(svg); removers.splice(0).forEach(remove => remove()); }
  load(); return { load, render: renderAll, destroy };
}

if (typeof document !== "undefined" && document.getElementById("atlas")) startAtlas();
