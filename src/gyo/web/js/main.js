import { fetchAtlas } from "./api.js";
import { fitTerritories } from "./atlas-layout.js";
import { createState, parentPrefix, prefixKey, selectNode, setSampleMode } from "./atlas-model.js";
import { renderInspector, renderMap } from "./atlas-render.js";

export function createRequestGuard() {
  let id = 0, controller;
  return { next() { controller?.abort(); controller = new AbortController(); return { id: ++id, signal: controller.signal }; }, isCurrent(candidate) { return candidate === id; } };
}

export function zoomView(view, factor, point) {
  const width = view.width / factor, height = view.height / factor;
  const rx = (point.x - view.x) / view.width, ry = (point.y - view.y) / view.height;
  return { x: point.x - rx * width, y: point.y - ry * height, width, height };
}

export function startAtlas(doc = document, win = window) {
  const svg = doc.getElementById("atlas"), inspector = doc.getElementById("inspector"), loading = doc.getElementById("mapLoading"), error = doc.getElementById("mapError");
  const status = doc.getElementById("projectionStatus"), crumbs = doc.getElementById("breadcrumbs"), back = doc.getElementById("backBtn");
  const cache = new Map(), guard = createRequestGuard();
  let state, placements = [], successful = false, currentPrefix = "root", view, dragging;
  const selectedNode = () => state?.payload.children.find(node => prefixKey(node.prefix) === prefixKey(state.selected || []));
  function bounds() { const rect = svg.getBoundingClientRect(); return { width: Math.max(1, rect.width || svg.clientWidth || 800), height: Math.max(1, rect.height || svg.clientHeight || 600) }; }
  function resetView() { const { width, height } = bounds(); view = { x: 0, y: 0, width, height }; svg.setAttribute("viewBox", `0 0 ${width} ${height}`); }
  function showStatus() { const projection = state.payload.projection || {}; const fallback = placements.some(item => item.layoutMode === "grid-fallback"); const parts = []; if (Number.isFinite(projection.stress)) parts.push(`Stress ${projection.stress.toFixed(3)}`); if (projection.warning) parts.push("projection warning"); if (fallback) parts.push("grid fallback distorts distances"); status.textContent = parts.join(" · "); status.classList.toggle("warning", Boolean(projection.warning || fallback)); }
  function renderBreadcrumbs() { crumbs.replaceChildren(); const focus = state.focus; for (let i = 0; i <= focus.length; i++) { const prefix = focus.slice(0, i), button = doc.createElement("button"); button.type = "button"; button.textContent = i ? String(focus[i - 1]) : "Root"; if (i === focus.length) button.setAttribute("aria-current", "page"); else button.addEventListener("click", () => load(prefixKey(prefix))); crumbs.appendChild(button); if (i < focus.length) crumbs.append("›"); } back.disabled = focus.length === 0; }
  function renderAll(reflow = true) { if (!state) return; if (reflow) { const { width, height } = bounds(); placements = fitTerritories(state.payload.children, width, height); resetView(); } renderMap(svg, placements, state, { select(node) { state = selectNode(state, node.prefix); renderAll(false); }, enter(node) { if (node.has_children) load(prefixKey(node.prefix)); } }); renderInspector(inspector, selectedNode(), state.sampleMode, { focus: state.payload.focus, mode(mode) { state = setSampleMode(state, mode); renderAll(false); }, enter(node) { if (node.has_children) load(prefixKey(node.prefix)); } }); renderBreadcrumbs(); showStatus(); }
  async function load(prefix = "root", force = false) { currentPrefix = prefix; const request = guard.next(); loading.hidden = false; error.hidden = true; try { const payload = !force && cache.has(prefix) ? cache.get(prefix) : await fetchAtlas(prefix, { signal: request.signal }); if (!guard.isCurrent(request.id)) return; cache.set(prefix, payload); state = createState(payload); successful = true; renderAll(); } catch (reason) { if (reason?.name === "AbortError" || !guard.isCurrent(request.id)) return; error.querySelector("p").textContent = reason instanceof Error ? reason.message : "Unable to load atlas"; error.hidden = false; if (!successful) svg.replaceChildren(); } finally { if (guard.isCurrent(request.id)) loading.hidden = true; } }
  error.querySelector("button").addEventListener("click", () => load(currentPrefix, true)); back.addEventListener("click", () => state && load(prefixKey(parentPrefix(state.focus)))); doc.getElementById("resetViewBtn").addEventListener("click", resetView);
  win.addEventListener("keydown", event => { if (event.key === "Escape" && state?.focus.length) load(prefixKey(parentPrefix(state.focus))); });
  let resizeTimer; win.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => renderAll(true), 100); });
  svg.addEventListener("wheel", event => { event.preventDefault(); const rect = svg.getBoundingClientRect(), point = { x: view.x + (event.clientX - rect.left) / rect.width * view.width, y: view.y + (event.clientY - rect.top) / rect.height * view.height }; view = zoomView(view, event.deltaY < 0 ? 1.15 : 1 / 1.15, point); svg.setAttribute("viewBox", `${view.x} ${view.y} ${view.width} ${view.height}`); }, { passive: false });
  svg.addEventListener("pointerdown", event => { dragging = { x: event.clientX, y: event.clientY, view: { ...view } }; svg.setPointerCapture?.(event.pointerId); }); svg.addEventListener("pointermove", event => { if (!dragging) return; const rect = svg.getBoundingClientRect(); view.x = dragging.view.x - (event.clientX - dragging.x) / rect.width * view.width; view.y = dragging.view.y - (event.clientY - dragging.y) / rect.height * view.height; svg.setAttribute("viewBox", `${view.x} ${view.y} ${view.width} ${view.height}`); }); svg.addEventListener("pointerup", () => { dragging = null; });
  load(); return { load, render: renderAll };
}

if (typeof document !== "undefined" && document.getElementById("atlas")) startAtlas();
