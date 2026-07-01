export const WIN = 3;
export const PEEK = 24;
export const SPINE_W = 18;
export const DEAD_W = 12;
export const BRANCH_COLORS = ["#4a9eff","#00d4aa","#ff6b6b","#ffd166","#c792ea","#f78c6c","#7fdbca","#82aaff","#f48fb1","#a5d6a7"];

export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

const isSliver = (n, collapsed) => (collapsed.has(n) && !n.dead) || n.dead;

export function computeLayout(focus, viewportW, { collapsedSet }) {
  const placed = [];
  (function layout(nodeN, x, w, depth, branch) {
    const spine = collapsedSet.has(nodeN) && nodeN !== focus;
    placed.push({ node: nodeN, x, w, depth, branch, spine });
    if (spine || nodeN.leaf || !nodeN.children || !nodeN.children.length) return;

    let nExpanded = 0, fixed = 0;
    for (const c of nodeN.children) {
      if (c.dead) fixed += DEAD_W;
      else if (collapsedSet.has(c)) fixed += SPINE_W;
      else nExpanded++;
    }
    const GAP = Math.max(2, 18 - depth * 6), SMALL = 2;
    const ordered = nodeN.children.slice().sort(
      (a, b) => ((collapsedSet.has(a) && !a.dead) ? 0 : 1) - ((collapsedSet.has(b) && !b.dead) ? 0 : 1)
    );
    const gapBetween = (a, b) => (isSliver(a, collapsedSet) || isSliver(b, collapsedSet)) ? SMALL : GAP;
    let gaps = 0;
    for (let i = 0; i < ordered.length - 1; i++) gaps += gapBetween(ordered[i], ordered[i + 1]);
    const avail = Math.max(0, w - fixed - gaps);
    const eachW = nExpanded ? avail / nExpanded : 0;
    let cx = x;
    ordered.forEach((c, i) => {
      const cw = c.dead ? DEAD_W : collapsedSet.has(c) ? SPINE_W : eachW;
      const cb = depth === 0 ? BRANCH_COLORS[nodeN.children.indexOf(c) % BRANCH_COLORS.length] : branch;
      layout(c, cx, cw, depth + 1, cb);
      cx += cw + (i < ordered.length - 1 ? gapBetween(c, ordered[i + 1]) : 0);
    });
  })(focus, 0, viewportW, 0, null);
  return placed;
}

export const scrollBounds = (maxWinTop, rowH, peek = PEEK) => ({ max: peek, min: peek - maxWinTop * rowH });
export const winTopFromScrollPx = (scrollPx, rowH, peek, maxWinTop) =>
  clamp(Math.round((peek - scrollPx) / rowH), 0, maxWinTop);

export function snapTarget(scrollPx, rowH, peek, maxWinTop) {
  const winTop = winTopFromScrollPx(scrollPx, rowH, peek, maxWinTop);
  return { winTop, scrollPx: peek - winTop * rowH };
}
