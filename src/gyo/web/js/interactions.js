/**
 * interactions — click/dblclick timer disambiguation + tooltip.
 * Ported from prototypes/mock-vi.html lines 440–519.
 */

/**
 * Create click/dblclick handlers with timer disambiguation.
 * Single click fires `onCollapse` after `delay` ms; double click cancels
 * it and fires `onDrill` immediately.
 */
export function makeClickHandlers({ onCollapse, onDrill, delay = 220 }) {
  let timer = null;
  return {
    onClick(e, node) {
      e.stopPropagation();
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => { onCollapse(node); timer = null; }, delay);
    },
    onDblClick(e, node) {
      e.stopPropagation();
      if (timer) { clearTimeout(timer); timer = null; }
      onDrill(node);
    },
  };
}

/**
 * Show tooltip near the cursor.
 */
export function showTip(e, node, tipEl, opts = {}) {
  const { residualColor, norm } = opts;
  const col = node.dead ? "#3a4350" : (residualColor ? residualColor(norm ? norm(node.residual ?? 0) : (node.residual_norm ?? 0)) : "#888");
  const prefix = node.isRoot ? "root" : "c" + (node.prefix ? node.prefix.join(",") : node.code);
  tipEl.style.display = "block";
  tipEl.innerHTML = `
    <div class="t-title">${prefix}</div>
    <div class="t-row"><span>occupancy</span><b>${node.occ ?? "—"}</b></div>
    <div class="t-row"><span>resíduo</span><b><span class="dot" style="background:${col}"></span>${node.dead ? "—" : (node.residual ?? 0).toFixed(3)}</b></div>
    <div class="t-row"><span>purity</span><b>${node.dead ? "—" : ((node.purity ?? 0) * 100).toFixed(0) + "%"}</b></div>
    ${node.leaf && !node.dead ? `<div class="t-row"><span>label</span><b>${node.label || "—"}</b></div>` : ""}
    ${node.dead ? `<div class="t-row"><span>status</span><b>dead codeword</b></div>` : ""}
  `;
  const pad = 14;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + 210 > innerWidth) x = e.clientX - 210;
  if (y + 120 > innerHeight) y = e.clientY - 120;
  tipEl.style.left = x + "px";
  tipEl.style.top = y + "px";
}

/**
 * Hide tooltip.
 */
export function hideTip(tipEl) {
  tipEl.style.display = "none";
}
