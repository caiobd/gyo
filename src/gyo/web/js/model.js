export const residualColor = (t) => {
  t = Math.max(0, Math.min(1, t));
  const g=[0,212,170], y=[255,234,167], r=[255,107,107];
  let a,b,u;
  if (t<0.5){ a=g; b=y; u=t/0.5; } else { a=y; b=r; u=(t-0.5)/0.5; }
  u = u < 0.004 ? 0 : u > 0.996 ? 1 : u;
  const c = a.map((v,i)=>Math.round(v+(b[i]-v)*u));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
};

export function buildModel(tree) {
  const byKey = new Map();
  const root = { code: null, prefix: [], level: 0, children: [], isRoot: true };
  byKey.set("", root);
  for (const n of tree.nodes) {
    if (n.prefix.length === 0) { Object.assign(root, { occ: n.occupancy, residual: n.mean_residual, residual_norm: n.residual_norm, purity: n.purity }); continue; }
    byKey.set(n.prefix.join(","), {
      code: n.prefix[n.prefix.length - 1], prefix: n.prefix, level: n.level,
      occ: n.occupancy, residual: n.mean_residual, residual_norm: n.residual_norm,
      purity: n.purity, leaf: n.level === tree.num_levels, children: [],
      dead: n.occupancy === 0 || n.is_dead === true,
      ...(n.label != null ? { label: n.label } : {}),
    });
  }
  for (const n of tree.nodes) {
    if (n.prefix.length === 0) continue;
    const me = byKey.get(n.prefix.join(","));
    const parent = byKey.get(n.prefix.slice(0, -1).join(","));
    me.parent = parent; parent.children.push(me);
  }
  return root;
}

export const depthOf = (n) => (!n.children || n.leaf || !n.children.length) ? 0
  : (() => { const alive = n.children.filter(c => !c.dead).map(depthOf); return alive.length ? 1 + Math.max(...alive) : 0; })();

export const prefixKey = (n) => n.isRoot ? "root" : n.prefix.join(",");
