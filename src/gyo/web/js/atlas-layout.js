const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

function validate(nodes, width, height) {
  if (!Array.isArray(nodes)) throw new TypeError("nodes must be an array");
  if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) {
    throw new RangeError("viewport dimensions must be positive finite numbers");
  }
  for (const node of nodes) {
    if (!node || !Array.isArray(node.position) || node.position.length !== 2 ||
        node.position.some(value => !Number.isFinite(value) || value < -1 || value > 1)) {
      throw new TypeError("node position must contain two finite values in [-1, 1]");
    }
    if (!Number.isFinite(node.occupancy) || node.occupancy < 0) {
      throw new RangeError("node occupancy must be a non-negative finite number");
    }
  }
}

function keepInBounds(circle, width, height, padding) {
  const inset = circle.r + padding;
  circle.cx = clamp(circle.cx, inset, width - inset);
  circle.cy = clamp(circle.cy, inset, height - inset);
}

function relax(circles, width, height, padding) {
  const tolerance = 1e-6;
  for (let pass = 0; pass < 80; pass++) {
    let overlap = false;
    for (let i = 0; i < circles.length; i++) {
      for (let j = i + 1; j < circles.length; j++) {
        const a = circles[i], b = circles[j];
        let dx = b.cx - a.cx, dy = b.cy - a.cy;
        let distance = Math.hypot(dx, dy);
        const required = a.r + b.r;
        if (distance >= required - tolerance) continue;
        overlap = true;
        if (distance < tolerance) {
          const angle = ((i + 1) * 2.399963 + (j + 1) * 1.618034) % (Math.PI * 2);
          dx = Math.cos(angle);
          dy = Math.sin(angle);
          distance = 1;
        }
        const push = (required - distance + tolerance) / 2;
        const ux = dx / distance, uy = dy / distance;
        a.cx -= ux * push; a.cy -= uy * push;
        b.cx += ux * push; b.cy += uy * push;
        keepInBounds(a, width, height, padding);
        keepInBounds(b, width, height, padding);
      }
    }
    if (!overlap) return true;
  }
  return separated(circles);
}

function separated(circles) {
  return circles.every((a, i) => circles.slice(i + 1).every(b =>
    Math.hypot(a.cx - b.cx, a.cy - b.cy) >= a.r + b.r - 0.01));
}

function gridFallback(nodes, width, height, padding) {
  const usableWidth = width - 2 * padding;
  const usableHeight = height - 2 * padding;
  const columns = Math.max(1, Math.ceil(Math.sqrt(nodes.length * usableWidth / usableHeight)));
  const rows = Math.ceil(nodes.length / columns);
  const cellWidth = usableWidth / columns;
  const cellHeight = usableHeight / rows;
  const cellRadius = Math.min(cellWidth, cellHeight) * 0.48;
  const maxOccupancy = Math.max(1, ...nodes.map(node => node.occupancy));
  const minFactor = 0.45;
  const order = nodes.map((node, index) => ({ node, index })).sort((a, b) =>
    a.node.position[1] - b.node.position[1] ||
    a.node.position[0] - b.node.position[0] ||
    a.index - b.index);
  const placed = new Array(nodes.length);
  order.forEach(({ node, index }, slot) => {
    const column = slot % columns;
    const row = Math.floor(slot / columns);
    placed[index] = {
      ...node,
      cx: padding + (column + 0.5) * cellWidth,
      cy: padding + (row + 0.5) * cellHeight,
      r: cellRadius * (minFactor + (1 - minFactor) * Math.sqrt(node.occupancy / maxOccupancy)),
      layoutMode: "grid-fallback",
    };
  });
  return placed;
}

/**
 * Fits normalized MDS positions into a viewport. Radius follows sqrt(occupancy)
 * with a minimum legibility radius; it is intentionally not exact area scaling.
 */
export function fitTerritories(nodes, width, height) {
  validate(nodes, width, height);
  if (nodes.length === 0) return [];

  const scale = Math.min(width, height);
  const padding = Math.min(12, scale * 0.02);
  if (nodes.length > 64) return gridFallback(nodes, width, height, padding);
  const maxOccupancy = Math.max(1, ...nodes.map(node => node.occupancy));
  const initial = nodes.map(node => ({
    ...node,
    cx: padding + (node.position[0] + 1) / 2 * (width - 2 * padding),
    cy: padding + (node.position[1] + 1) / 2 * (height - 2 * padding),
    r: scale * (0.025 + 0.075 * Math.sqrt(node.occupancy / maxOccupancy)),
  }));

  let circles = initial;
  for (let attempt = 0; attempt < 6; attempt++) {
    circles.forEach(circle => keepInBounds(circle, width, height, padding));
    if (relax(circles, width, height, padding) && separated(circles)) return circles;
    circles = initial.map(circle => ({
      ...circle,
      r: circle.r * Math.pow(0.84, attempt + 1),
      cx: circle.cx,
      cy: circle.cy,
    }));
  }
  return gridFallback(nodes, width, height, padding);
}
