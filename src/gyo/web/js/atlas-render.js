import { thumbUrl } from "./api.js";
import { prefixKey } from "./atlas-model.js";

const SVG = "http://www.w3.org/2000/svg";
const prefixName = prefix => prefix?.length ? `Group ${prefix.join(".")}` : "Root group";
const samePrefix = (a, b) => Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => v === b[i]);
const svgEl = name => document.createElementNS(SVG, name);
const renderSessions = new WeakMap();

export function cancelMapInteractions(svg) {
  const session = renderSessions.get(svg);
  if (session?.clickTimer) clearTimeout(session.clickTimer);
  renderSessions.delete(svg);
}

function text(parent, value, className) {
  const element = document.createElement("span");
  if (className) element.className = className;
  element.textContent = String(value);
  parent.appendChild(element);
  return element;
}

export function renderMap(svg, placements, state, handlers = {}) {
  cancelMapInteractions(svg);
  const session = { clickTimer: null, activities: new Map() };
  renderSessions.set(svg, session);
  if (Number.isFinite(handlers.width) && handlers.width > 0 && Number.isFinite(handlers.height) && handlers.height > 0) {
    svg.setAttribute("viewBox", `0 0 ${handlers.width} ${handlers.height}`);
  }
  svg.replaceChildren();
  const defs = svgEl("defs");
  svg.appendChild(defs);
  const width = handlers.width || 800, height = handlers.height || 600;
  const margin = 12;
  const extents = placements.length ? {
    left: Math.min(...placements.map(node => node.cx - node.r)) - margin,
    top: Math.min(...placements.map(node => node.cy - node.r)) - margin,
    right: Math.max(...placements.map(node => node.cx + node.r)) + margin,
    bottom: Math.max(...placements.map(node => node.cy + node.r)) + margin,
  } : { left: width * .2, top: height * .2, right: width * .8, bottom: height * .8 };
  const bx = Math.max(4, extents.left), by = Math.max(4, extents.top), br = Math.min(width - 4, extents.right), bb = Math.min(height - 4, extents.bottom);
  const boundary = svgEl("rect"); boundary.classList.add("focus-boundary");
  boundary.dataset.prefix = prefixKey(state.focus || []); boundary.setAttribute("aria-hidden", "true");
  boundary.setAttribute("x", bx); boundary.setAttribute("y", by); boundary.setAttribute("width", Math.max(0, br - bx)); boundary.setAttribute("height", Math.max(0, bb - by)); boundary.setAttribute("rx", 18); svg.appendChild(boundary);
  const anchorX = (bx + br) / 2, anchorY = Math.min(bb - 8, by + 12);
  const links = svgEl("g"); links.classList.add("hierarchy-links"); links.setAttribute("aria-hidden", "true"); svg.appendChild(links);
  placements.forEach((node, index) => {
    const link = svgEl("line"); link.classList.add("hierarchy-link"); link.dataset.index = String(index);
    link.setAttribute("x1", anchorX); link.setAttribute("y1", anchorY); link.setAttribute("x2", node.cx); link.setAttribute("y2", node.cy); links.appendChild(link);
  });
  const anchor = svgEl("g"); anchor.classList.add("focus-anchor"); anchor.setAttribute("aria-hidden", "true");
  const anchorCircle = svgEl("circle"); anchorCircle.setAttribute("cx", anchorX); anchorCircle.setAttribute("cy", anchorY); anchorCircle.setAttribute("r", 7);
  const anchorLabel = svgEl("text"); anchorLabel.setAttribute("x", anchorX + 12); anchorLabel.setAttribute("y", anchorY + 4); anchorLabel.textContent = prefixName(state.focus || []); anchor.append(anchorCircle, anchorLabel); svg.appendChild(anchor);
  if (!placements.length) { const empty = svgEl("text"); empty.classList.add("empty-level"); empty.setAttribute("x", width / 2); empty.setAttribute("y", height / 2); empty.textContent = "No child groups at this level"; svg.appendChild(empty); }
  const childrenGroup = svgEl("g"); childrenGroup.classList.add("children-group"); childrenGroup.setAttribute("role", "group"); childrenGroup.setAttribute("aria-label", `Children of ${prefixName(state.focus || [])}`); svg.appendChild(childrenGroup);
  const selectedIndex = placements.findIndex(node => samePrefix(state.selected, node.prefix));
  placements.forEach((node, index) => {
    const group = svgEl("g");
    group.classList.add("territory");
    if (node.aggregate) group.classList.add("aggregate");
    group.dataset.prefix = node.aggregate ? "aggregate" : prefixKey(node.prefix);
    if (samePrefix(state.selected, node.prefix)) group.classList.add("selected");
    group.setAttribute("role", "treeitem");
    group.setAttribute("aria-level", String((state.focus?.length || 0) + 1)); group.setAttribute("aria-posinset", String(index + 1)); group.setAttribute("aria-setsize", String(placements.length));
    if (node.aggregate) group.setAttribute("aria-expanded", "false");
    group.setAttribute("tabindex", String(index === (selectedIndex < 0 ? 0 : selectedIndex) ? 0 : -1));
    group.setAttribute("aria-selected", String(samePrefix(state.selected, node.prefix)));
    group.setAttribute("aria-label", node.aggregate ? (node.revealable === false ? `${node.count} groups hidden — enter a branch or collapse` : `${node.count} more groups, Reveal more groups`) : `${prefixName(node.prefix)}, ${node.occupancy} items`);
    const circle = svgEl("circle");
    circle.setAttribute("cx", node.cx); circle.setAttribute("cy", node.cy); circle.setAttribute("r", node.r);
    group.appendChild(circle);
    const clip = svgEl("clipPath");
    const clipId = `territory-clip-${index}`;
    clip.id = clipId;
    const clipCircle = circle.cloneNode();
    clip.appendChild(clipCircle); defs.appendChild(clip);
    const samples = node.samples?.representative || [];
    samples.slice(0, 4).forEach((sample, sampleIndex) => {
      const image = svgEl("image");
      const size = Math.max(24, node.r * .58);
      const col = sampleIndex % 2, row = Math.floor(sampleIndex / 2);
      image.setAttribute("href", thumbUrl(sample.idx));
      image.setAttribute("x", node.cx - size + col * size);
      image.setAttribute("y", node.cy - size + row * size);
      image.setAttribute("width", size); image.setAttribute("height", size);
      image.setAttribute("preserveAspectRatio", "xMidYMid slice"); image.setAttribute("clip-path", `url(#${clipId})`);
      image.setAttribute("visibility", "visible");
      image.addEventListener("error", () => {
        image.setAttribute("visibility", "hidden");
        const retry = svgEl("g"); retry.classList.add("svg-image-retry"); retry.setAttribute("role", "button"); retry.setAttribute("tabindex", "0"); retry.setAttribute("aria-label", `Retry sample ${sample.idx}`);
        const plate = svgEl("rect"); plate.setAttribute("x", image.getAttribute("x")); plate.setAttribute("y", image.getAttribute("y")); plate.setAttribute("width", size); plate.setAttribute("height", size);
        const retryText = svgEl("text"); retryText.setAttribute("x", Number(image.getAttribute("x")) + size / 2); retryText.setAttribute("y", Number(image.getAttribute("y")) + size / 2); retryText.textContent = "Retry";
        retry.append(plate, retryText); group.appendChild(retry);
        const reload = event => { event.stopPropagation(); retry.remove(); image.setAttribute("visibility", "visible"); image.setAttribute("href", ""); image.setAttribute("href", thumbUrl(sample.idx)); };
        retry.addEventListener("click", reload); retry.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") reload(event); });
      });
      group.appendChild(image);
    });
    const label = svgEl("text"); label.setAttribute("x", node.cx); label.setAttribute("y", node.cy - node.r + 18);
    label.textContent = node.aggregate ? node.label : prefixName(node.prefix); label.classList.add("territory-label"); group.appendChild(label);
    const occupancy = svgEl("text"); occupancy.setAttribute("x", node.cx); occupancy.setAttribute("y", node.cy + node.r - 10);
    occupancy.textContent = `${node.occupancy} items`; occupancy.classList.add("territory-count"); group.appendChild(occupancy);
    const activity = { hover: false, focus: false };
    const setPath = (kind, active) => {
      activity[kind] = active; session.activities.set(index, activity);
      const activeIndices = [...session.activities].filter(([, value]) => value.hover || value.focus).map(([key]) => key);
      const anyActive = activeIndices.length > 0, primary = activeIndices.at(-1);
      childrenGroup.querySelectorAll(".territory").forEach((item, itemIndex) => item.classList.toggle("is-path", activeIndices.includes(itemIndex)));
      childrenGroup.querySelectorAll(".territory:not(.aggregate)").forEach((item, itemIndex) => item.classList.toggle("is-sibling", anyActive && !activeIndices.includes(itemIndex)));
      [...links.children].forEach((link, linkIndex) => link.classList.toggle("is-path", activeIndices.includes(linkIndex)));
      anchor.classList.toggle("is-path", anyActive); svg.classList.toggle("has-active-path", anyActive);
      handlers.path?.(anyActive ? placements[primary].prefix : null);
    };
    group.addEventListener("pointerenter", () => setPath("hover", true)); group.addEventListener("pointerleave", () => setPath("hover", false));
    group.addEventListener("focus", () => setPath("focus", true)); group.addEventListener("blur", () => setPath("focus", false));
    group.addEventListener("click", () => { if (node.aggregate) { handlers.expand?.(); return; } clearTimeout(session.clickTimer); session.clickTimer = setTimeout(() => { session.clickTimer = null; handlers.select?.(node); }, 180); });
    group.addEventListener("dblclick", event => { if (!node.has_children) return; clearTimeout(session.clickTimer); session.clickTimer = null; event.preventDefault(); handlers.enter?.(node); });
    group.addEventListener("keydown", event => {
      if (node.aggregate && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); handlers.expand?.(); return; }
      if (event.key === "Enter") { event.preventDefault(); handlers.select?.(node); }
      if (event.key === " " && node.has_children) { event.preventDefault(); handlers.enter?.(node); }
      const territories = [...svg.querySelectorAll('[role="treeitem"]')];
      const current = territories.indexOf(group);
      let target = null;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") target = territories[(current + 1) % territories.length];
      if (event.key === "ArrowLeft" || event.key === "ArrowUp") target = territories[(current - 1 + territories.length) % territories.length];
      if (event.key === "Home") target = territories[0];
      if (event.key === "End") target = territories.at(-1);
      if (target) { event.preventDefault(); territories.forEach(item => item.setAttribute("tabindex", item === target ? "0" : "-1")); target.focus(); }
    });
    childrenGroup.appendChild(group);
  });
}

function metric(grid, label, value, format = String) {
  if (value === null || value === undefined || !Number.isFinite(value)) return;
  const item = document.createElement("div"); item.className = "metric";
  text(item, label, "metric-label"); text(item, format(value), "metric-value"); grid.appendChild(item);
}

function sampleGrid(parent, samples) {
  const grid = document.createElement("div"); grid.className = "sample-grid"; parent.appendChild(grid);
  if (!samples?.length) { text(grid, "—", "empty-samples"); return; }
  samples.forEach(sample => {
    const figure = document.createElement("figure");
    const slot = document.createElement("div"); slot.className = "thumb-slot";
    const load = () => {
      slot.replaceChildren();
      const image = document.createElement("img"); image.src = thumbUrl(sample.idx); image.alt = sample.label == null ? `Sample ${sample.idx}` : String(sample.label); image.loading = "lazy";
      image.addEventListener("error", () => { slot.replaceChildren(); text(slot, "Image unavailable", "thumb-error"); const retry = document.createElement("button"); retry.type = "button"; retry.textContent = "Retry"; retry.addEventListener("click", load); slot.appendChild(retry); });
      slot.appendChild(image);
    };
    load(); figure.appendChild(slot); text(figure, sample.label ?? `Sample ${sample.idx}`, "sample-label"); grid.appendChild(figure);
  });
}

export function renderInspector(container, node, mode, handlers = {}) {
  container.replaceChildren();
  if (!node) { text(container, "Select a territory to inspect it.", "inspector-empty"); return; }
  const heading = document.createElement("h2"); heading.textContent = prefixName(node.prefix); container.appendChild(heading);
  const metrics = document.createElement("div"); metrics.className = "metrics"; container.appendChild(metrics);
  metric(metrics, "Occupancy", node.occupancy, value => value.toLocaleString());
  metric(metrics, "Purity", node.purity, value => `${(value * 100).toFixed(1)}%`);
  metric(metrics, "Residual", node.mean_residual, value => value.toFixed(3));
  metric(metrics, "Parent distance", node.parent_distance, value => value.toFixed(3));
  metric(metrics, "Token norm", node.token_norm, value => value.toFixed(3));
  const tabs = document.createElement("div"); tabs.className = "tabs"; tabs.setAttribute("aria-label", "Sample mode");
  [["representative", "Representative"], ["outliers", "Outliers"], ["parent", "Parent comparison"]].forEach(([key, label]) => {
    const button = document.createElement("button"); button.type = "button"; button.textContent = label; button.setAttribute("aria-pressed", String(mode === key)); button.addEventListener("click", () => handlers.mode?.(key)); tabs.appendChild(button);
  });
  container.appendChild(tabs);
  if (node.has_children) { const enter = document.createElement("button"); enter.type = "button"; enter.className = "enter-group"; enter.textContent = "Enter group"; enter.addEventListener("click", () => handlers.enter?.(node)); container.appendChild(enter); }
  if (mode === "parent") {
    const comparison = document.createElement("div"); comparison.className = "comparison"; container.appendChild(comparison);
    const parent = document.createElement("section"); const parentTitle = document.createElement("h3"); parentTitle.textContent = "Current focus"; parent.appendChild(parentTitle); sampleGrid(parent, handlers.focus?.samples?.representative); comparison.appendChild(parent);
    const child = document.createElement("section"); const childTitle = document.createElement("h3"); childTitle.textContent = "Selected group"; child.appendChild(childTitle); sampleGrid(child, node.samples?.representative); comparison.appendChild(child);
  } else sampleGrid(container, node.samples?.[mode]);
}
