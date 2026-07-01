/**
 * window.js — sliding 3-level window controller.
 * Ported from prototypes/mock-vi.html lines 265–311, 472–548.
 *
 * Owns scrollPx/winTop, wheel free-follow+snap, keys, rail, peek/fades, wrapper transform.
 * Sliding does NOT re-render the icicle — only the wrapper transform + rail.
 */
import { PEEK, WIN, clamp, snapTarget, scrollBounds } from "./layout.js";

export function createWindowController({ scroller, icicle, railEl, winStatEl, getMaxDepth, getRowH, onWinTopChange }) {
  let winTop = 1;
  let scrollPx = PEEK;
  let snapTimer = null;

  function maxWinTop() { return Math.max(0, getMaxDepth() - WIN + 1); }
  const scrollMax = () => PEEK;
  const scrollMin = () => PEEK - maxWinTop() * getRowH();

  function setScrollPx(px, animate) {
    scrollPx = clamp(px, scrollMin(), scrollMax());
    scroller.style.transition = animate ? "transform .22s ease" : "none";
    scroller.style.transform = `translateY(${scrollPx}px)`;
  }

  function setWinTop(nv, animate) {
    winTop = clamp(nv, 0, maxWinTop());
    setScrollPx(PEEK - winTop * getRowH(), animate);
    renderRail();
    updateWinStat();
    if (onWinTopChange) onWinTopChange(winTop);
  }

  function slide(d) { setWinTop(winTop + d, true); }

  function renderRail() {
    if (!railEl) return;
    railEl.innerHTML = "";
    const md = getMaxDepth();
    for (let d = 0; d <= md; d++) {
      const t = document.createElement("div");
      t.className = "rail-tick" + (d >= winTop && d < winTop + WIN ? " active" : "");
      t.textContent = d;
      t.title = "nível " + d;
      t.addEventListener("click", () => setWinTop(d, true));
      railEl.appendChild(t);
    }
  }

  function updateWinStat() {
    if (!winStatEl) return;
    const md = getMaxDepth();
    winStatEl.textContent = `${winTop}–${Math.min(winTop + WIN - 1, md)} / ${md}`;
  }

  function syncToRender(newWinTop) {
    winTop = clamp(newWinTop ?? winTop, 0, maxWinTop());
    setScrollPx(PEEK - winTop * getRowH(), false);
    renderRail();
    updateWinStat();
  }

  // wheel free-follow + snap
  function onWheel(e) {
    e.preventDefault();
    setScrollPx(scrollPx - e.deltaY, false);
    winTop = clamp(Math.round((PEEK - scrollPx) / getRowH()), 0, maxWinTop());
    renderRail();
    updateWinStat();
    if (snapTimer) clearTimeout(snapTimer);
    snapTimer = setTimeout(() => {
      const s = snapTarget(scrollPx, getRowH(), PEEK, maxWinTop());
      winTop = s.winTop;
      setScrollPx(s.scrollPx, true);
      renderRail();
      updateWinStat();
      if (onWinTopChange) onWinTopChange(winTop);
    }, 130);
  }

  // arrow keys
  function onKeyDown(e) {
    if (e.key === "ArrowDown") { e.preventDefault(); slide(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); slide(-1); }
  }

  // bind events
  if (icicle) icicle.addEventListener("wheel", onWheel, { passive: false });
  addEventListener("keydown", onKeyDown);

  return { setWinTop, slide, syncToRender, renderRail, updateWinStat };
}
