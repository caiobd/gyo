import { fetchTree, fetchNodeItems, thumbUrl } from "./api.js";
import { buildModel, depthOf, residualColor, prefixKey } from "./model.js";
import { computeLayout, PEEK, WIN } from "./layout.js";
import { renderPlacements, fillMosaic } from "./render.js";
import { loadPrefs, savePrefs } from "./persist.js";
import { makeClickHandlers, showTip, hideTip } from "./interactions.js";
import { createWindowController } from "./window.js";

const icicle = document.getElementById("icicle");
const scroller = document.getElementById("scroller");
const modeBtn = document.getElementById("modeBtn");
const crumbsEl = document.getElementById("crumbs");
const tipEl = document.getElementById("tip");
const railEl = document.getElementById("rail");
const winStatEl = document.getElementById("winstat");
let root, focus, collapsedSet = new Set();
let mode = loadPrefs().mode || "images";
let treeJson, winCtrl;

function toggleMode() {
  mode = mode === "images" ? "residual" : "images";
  savePrefs({ mode });
  updateModeBtn();
  render();
}

function updateModeBtn() {
  if (modeBtn) modeBtn.textContent = "modo: " + (mode === "images" ? "imagens" : "resíduo");
}

function setFocus(node) {
  if (node.dead) return;
  if (node.isRoot && focus.isRoot) return;
  focus = node;
  render();
}

function renderCrumbs() {
  if (!crumbsEl) return;
  crumbsEl.innerHTML = "";
  const path = [];
  let n = focus;
  while (n) { path.unshift(n); n = n.parent; }
  path.forEach((node, i) => {
    if (i > 0) {
      const s = document.createElement("span");
      s.className = "crumb-sep";
      s.textContent = "›";
      crumbsEl.appendChild(s);
    }
    const c = document.createElement("span");
    c.className = "crumb" + (node === focus ? " active" : "");
    c.textContent = node.isRoot ? "root" : "c" + node.code;
    if (node !== focus) c.addEventListener("click", () => setFocus(node));
    crumbsEl.appendChild(c);
  });
}

const clickHandlers = makeClickHandlers({
  onCollapse(node) {
    if (node.isRoot) return;
    collapsedSet.add(node);
    render();
  },
  onDrill(node) {
    if (!node.leaf) setFocus(node);
  },
});

let currentMaxDepth = 0, currentRowH = 70;

async function boot() {
  treeJson = await fetchTree();
  root = buildModel(treeJson);
  focus = root;
  const savedWinTop = loadPrefs().winTop;
  winCtrl = createWindowController({
    scroller, icicle, railEl, winStatEl,
    getMaxDepth: () => currentMaxDepth,
    getRowH: () => currentRowH,
    onWinTopChange(wt) { savePrefs({ winTop: wt }); },
  });
  updateModeBtn();
  document.getElementById("dead").textContent = JSON.stringify(treeJson.dead_codewords || []);
  render();
  if (savedWinTop != null) winCtrl.syncToRender(savedWinTop);
  else winCtrl.syncToRender(0);
}

function render() {
  const W = icicle.clientWidth, H = icicle.clientHeight;
  currentMaxDepth = depthOf(focus);
  currentRowH = Math.max(70, (H - 2 * PEEK) / WIN);
  scroller.style.height = (currentMaxDepth + 1) * currentRowH + "px";
  const placements = computeLayout(focus, W, { collapsedSet });
  renderPlacements(scroller, placements, {
    mode, ROW_H: currentRowH, maxDepth: currentMaxDepth, residualColor, thumbUrl,
    onClick: clickHandlers.onClick,
    onDblClick: clickHandlers.onDblClick,
    onExpand(e, node) {
      collapsedSet.delete(node);
      render();
    },
    onHover(e, node) { showTip(e, node, tipEl, { residualColor }); },
    onLeave() { hideTip(tipEl); },
    onFillMosaic(el, node) {
      if (mode !== "images") return;
      fetchNodeItems(prefixKey(node)).then(data => {
        if (data && data.items) fillMosaic(el, data.items, { w: parseFloat(el.style.width), ROW_H: currentRowH, depth: placements.find(p => p.node === node)?.depth ?? 0, thumbUrl });
      });
    },
  });
  renderCrumbs();
}

let rT;
addEventListener("resize", () => { clearTimeout(rT); rT = setTimeout(render, 120); });
addEventListener("keydown", e => {
  if (e.key.toLowerCase() === "c") toggleMode();
});
if (modeBtn) modeBtn.addEventListener("click", toggleMode);
boot();
