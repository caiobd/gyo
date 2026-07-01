/**
 * renderPlacements — icicle renderer with mosaic support.
 * Ported from prototypes/mock-vi.html lines 360–465.
 *
 * @param {HTMLElement} scroller  - the #scroller div
 * @param {Array}       placements - from computeLayout()
 * @param {Object}      ctx
 *   ctx.mode           - "residual" | "images"
 *   ctx.ROW_H          - height of one level in px
 *   ctx.maxDepth       - deepest depth under focus
 *   ctx.residualColor  - (t:0..1) => css color string
 *   ctx.onFillMosaic   - (el, node) => void
 *   ctx.thumbUrl       - (idx) => url string
 *   ctx.onClick        - (e, node) => void  — single click on expanded node
 *   ctx.onDblClick     - (e, node) => void  — double click on expanded node
 *   ctx.onExpand       - (e, node) => void  — click on spine to expand
 *   ctx.onHover        - (e, node) => void  — mouseenter on any node
 *   ctx.onLeave        - (e) => void        — mouseleave on any node
 */

/**
 * Deterministic even-stride sub-sample: pick `slots` items spread uniformly
 * across the input array. Returns a (possibly shorter) copy when items.length <= slots.
 */
export const sampleToSlots = (items, slots) => {
  if (items.length <= slots) return items.slice();
  const out = [];
  const step = items.length / slots;
  for (let s = 0; s < slots; s++) out.push(items[Math.min(items.length - 1, Math.floor(s * step))]);
  return out;
};

/**
 * Compute tile size for a mosaic given the node's inner dimensions.
 * @param {number} innerW - available width (px, after padding)
 * @param {number} innerH - available height (px, after label/chip)
 * @returns {number} tile size in px
 */
export const tileForBox = (innerW, innerH) =>
  Math.max(11, Math.min(34, Math.round(Math.sqrt(Math.max(0, innerW) * Math.max(0, innerH) / 8))));

/**
 * Fill an element with a sampled thumbnail mosaic.
 * @param {HTMLElement} el      - the node element
 * @param {Array}       items   - [{idx, path, label}, ...]
 * @param {Object}      opts
 *   opts.w         - node box width (px)
 *   opts.ROW_H     - row height (px)
 *   opts.depth     - node depth
 *   opts.thumbUrl  - (idx) => url string
 */
export function fillMosaic(el, items, opts) {
  const { w, ROW_H, depth, thumbUrl } = opts;
  const innerW = Math.max(0, w - 12);
  const innerH = Math.max(0, (ROW_H - 8) - 30);
  const tile = tileForBox(innerW, innerH);
  const gap = 3;
  const cols = Math.floor((innerW + gap) / (tile + gap));
  const rows = Math.floor((innerH + gap) / (tile + gap));
  const slots = Math.max(0, cols * rows);
  if (slots < 2 || !items || !items.length) return;

  const box = document.createElement("div");
  box.className = "thumbs";
  const sampled = sampleToSlots(items, slots);
  for (const it of sampled) {
    const t = document.createElement("div");
    t.className = "thumb";
    t.style.width = t.style.height = tile + "px";
    const img = document.createElement("img");
    img.src = thumbUrl(it.idx);
    img.alt = it.label || "";
    img.loading = "lazy";
    t.appendChild(img);
    box.appendChild(t);
  }
  el.appendChild(box);
}
export function renderPlacements(scroller, placements, ctx) {
  const { ROW_H, maxDepth, residualColor, onFillMosaic, onClick, onDblClick, onExpand, onHover, onLeave } = ctx;

  scroller.innerHTML = "";

  for (const p of placements) {
    const { node, x, w, depth } = p;
    const el = document.createElement("div");
    el.className = "node" + (node.dead ? " dead" : "") + (p.spine ? " spine" : "");
    el.style.left   = x + "px";
    el.style.width  = Math.max(w - 3, node.dead ? 10 : 6) + "px";
    el.style.top    = depth * ROW_H + "px";
    el.style.height = ((p.spine ? (maxDepth + 1 - depth) : 1) * ROW_H - 8) + "px";

    // branch top-cap (colored by branch identity)
    if (p.branch && !node.dead) {
      const cap = document.createElement("div");
      cap.className = "branchcap";
      cap.style.background = p.branch;
      el.appendChild(cap);
    }

    if (p.spine) {
      // spine: collapsed subtree — vertical label + occ chip
      el.style.background = residualColor(node.residual_norm ?? 0);
      const lab = document.createElement("div");
      lab.className = "spinelabel";
      lab.textContent = node.isRoot ? "root" : "c" + node.code;
      el.appendChild(lab);
      const occ = document.createElement("div");
      occ.className = "nocc";
      occ.textContent = node.occ;
      el.appendChild(occ);
      if (onExpand) el.addEventListener("click", e => onExpand(e, node));
      if (onHover) el.addEventListener("mousemove", e => onHover(e, node));
      if (onLeave) el.addEventListener("mouseleave", e => onLeave(e));
      scroller.appendChild(el);
      continue;
    }

    if (!node.dead) {
      el.style.background = residualColor(node.residual_norm ?? 0);

      // label chip
      const lab = document.createElement("div");
      lab.className = "nlabel";
      lab.textContent = (node.isRoot ? "root" : "c" + node.code) + (node.leaf && node.label ? " · " + node.label : "");
      el.appendChild(lab);

      // occupancy chip
      const occ = document.createElement("div");
      occ.className = "nocc";
      occ.textContent = node.occ;
      el.appendChild(occ);

      // mosaic hook (no-op PR3; filled in PR4)
      onFillMosaic(el, node);

      // purity bar
      const pur = document.createElement("div");
      pur.className = "purity";
      pur.style.width = ((node.purity ?? 0) * 100) + "%";
      el.appendChild(pur);

      if (!node.isRoot) el.style.cursor = "zoom-out";
      if (onClick) el.addEventListener("click", e => onClick(e, node));
      if (onDblClick) el.addEventListener("dblclick", e => onDblClick(e, node));
      if (onHover) el.addEventListener("mousemove", e => onHover(e, node));
      if (onLeave) el.addEventListener("mouseleave", e => onLeave(e));
    } else {
      // dead node label
      const lab = document.createElement("div");
      lab.className = "nlabel";
      lab.textContent = "c" + node.code + " ∅ dead";
      el.appendChild(lab);
      if (onHover) el.addEventListener("mousemove", e => onHover(e, node));
      if (onLeave) el.addEventListener("mouseleave", e => onLeave(e));
    }

    scroller.appendChild(el);
  }
}
