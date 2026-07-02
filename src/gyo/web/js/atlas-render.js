import { thumbUrl } from "./api.js";

const SVG = "http://www.w3.org/2000/svg";
const prefixName = prefix => prefix?.length ? `Group ${prefix.join(".")}` : "Root group";
const samePrefix = (a, b) => Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => v === b[i]);
const svgEl = name => document.createElementNS(SVG, name);

function text(parent, value, className) {
  const element = document.createElement("span");
  if (className) element.className = className;
  element.textContent = String(value);
  parent.appendChild(element);
  return element;
}

export function renderMap(svg, placements, state, handlers = {}) {
  svg.replaceChildren();
  const defs = svgEl("defs");
  svg.appendChild(defs);
  placements.forEach((node, index) => {
    const group = svgEl("g");
    group.classList.add("territory");
    if (samePrefix(state.selected, node.prefix)) group.classList.add("selected");
    group.setAttribute("role", "treeitem");
    group.setAttribute("tabindex", "0");
    group.setAttribute("aria-selected", String(samePrefix(state.selected, node.prefix)));
    group.setAttribute("aria-label", `${prefixName(node.prefix)}, ${node.occupancy} items`);
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
      group.appendChild(image);
    });
    const label = svgEl("text"); label.setAttribute("x", node.cx); label.setAttribute("y", node.cy - node.r + 18);
    label.textContent = prefixName(node.prefix); label.classList.add("territory-label"); group.appendChild(label);
    const occupancy = svgEl("text"); occupancy.setAttribute("x", node.cx); occupancy.setAttribute("y", node.cy + node.r - 10);
    occupancy.textContent = `${node.occupancy} items`; occupancy.classList.add("territory-count"); group.appendChild(occupancy);
    let clickTimer;
    group.addEventListener("click", () => { clearTimeout(clickTimer); clickTimer = setTimeout(() => handlers.select?.(node), 180); });
    group.addEventListener("dblclick", event => { clearTimeout(clickTimer); event.preventDefault(); if (node.has_children) handlers.enter?.(node); });
    group.addEventListener("keydown", event => {
      if (event.key === "Enter") { event.preventDefault(); handlers.select?.(node); }
      if (event.key === " " && node.has_children) { event.preventDefault(); handlers.enter?.(node); }
    });
    svg.appendChild(group);
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
  const tabs = document.createElement("div"); tabs.className = "tabs"; tabs.setAttribute("role", "tablist");
  [["representative", "Representative"], ["outliers", "Outliers"], ["parent", "Parent comparison"]].forEach(([key, label]) => {
    const button = document.createElement("button"); button.type = "button"; button.textContent = label; button.setAttribute("role", "tab"); button.setAttribute("aria-selected", String(mode === key)); button.addEventListener("click", () => handlers.mode?.(key)); tabs.appendChild(button);
  });
  container.appendChild(tabs);
  if (mode === "parent") {
    const comparison = document.createElement("div"); comparison.className = "comparison"; container.appendChild(comparison);
    const parent = document.createElement("section"); const parentTitle = document.createElement("h3"); parentTitle.textContent = "Current focus"; parent.appendChild(parentTitle); sampleGrid(parent, handlers.focus?.samples?.representative); comparison.appendChild(parent);
    const child = document.createElement("section"); const childTitle = document.createElement("h3"); childTitle.textContent = "Selected group"; child.appendChild(childTitle); sampleGrid(child, node.samples?.representative); comparison.appendChild(child);
  } else sampleGrid(container, node.samples?.[mode]);
  if (node.has_children) { const enter = document.createElement("button"); enter.type = "button"; enter.className = "enter-group"; enter.textContent = "Enter group"; enter.addEventListener("click", () => handlers.enter?.(node)); container.appendChild(enter); }
}
