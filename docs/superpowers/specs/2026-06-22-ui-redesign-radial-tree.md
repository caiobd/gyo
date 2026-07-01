# UI Redesign — Radial Tree + Dark Mode + Enhanced Panel

> **Spec for agentic workers:** This spec describes the visual/interaction improvements to the gyo RQ embedding inspector web UI.

## Goal

Transform the current basic tree visualization into a modern, polished radial tree with dark mode, animations, and an enhanced side panel showing node metrics and thumbnails.

## Current State

- Basic D3 tree layout (vertical, straight lines)
- Minimal CSS (9 lines), no dark mode
- Side panel shows only thumbnails on click
- No animations, no hover effects, no tooltips

## Target State

### Visual Design

**Color Palette (dark mode with accents):**
- Main background: `#0f0f1a` (very dark blue)
- Panel background: `#1a1a2e` (dark blue)
- Primary text: `#e0e0e0` (soft white)
- Secondary text: `#888` (gray)
- Primary accent: `#4a9eff` (bright blue)
- Secondary accent: `#00d4aa` (cyan)
- Warm accent: `#ff6b6b` (pink/red)
- Node glow: `rgba(74, 158, 255, 0.4)` (blue glow)

**Typography:**
- Font: `Inter` or `system-ui`
- Titles: 18-20px, font-weight 600
- Body: 14px, font-weight 400
- Data/numbers: 12px, monospace

**Layout:**
- Header compact with title + controls
- Tree radial occupies 70% width
- Side panel fixed at 30% right

### Tree Visualization (Radial)

**Layout:**
- Root at center with larger size (radius 20-25px)
- Children arranged in concentric rings
- Each level = one ring (radius increases by level)
- Leaf nodes on outer rings

**Node Styling:**
- Color based on `residual_norm` (blue → red scale)
- Dead nodes (occupancy=0): `#333` (dark gray)
- Hover: glow/luminous shadow effect

**Links:**
- Bézier curves between parent → child nodes
- Color: `#333` with 40% opacity
- Hover: primary accent color

**Interactions:**
- **Hover**: Floating tooltip with node name, occupancy, residual
- **Click**: Zoom to node (250ms transition)
- **Double-click**: Expand/collapse children
- **Scroll**: Zoom in/out

**Labels:**
- Internal nodes: no label (data in tooltip)
- Leaf nodes: external label when space allows
- Root: "root" label below node

### Side Panel

**Layout:**
- Panel header: "Bucket [prefix] — N items"
- Metrics: Ocupância, Residual Médio, Pureza (when available)
- Thumbnail grid: 4-5 columns, 48x48px with labels
- Vertical scroll for many items

**Metrics Display:**
- Ocupância: item count
- Residual: colored progress bar (blue→red)
- Pureza: percentage when labels available

### Animations

**Transitions (250ms ease-out):**
- Expand/collapse: children fade + slide in
- Zoom: smooth camera transition
- Tooltip: fast fade in (150ms)
- Hover: glow effect 150ms

**Scroll/Zoom:**
- Mouse wheel: zoom in/out centered on cursor
- Pan: drag on background

## Files to Modify

- `src/gyo/web/index.html` — Update structure, add dark mode classes
- `src/gyo/web/style.css` — Complete redesign with dark theme
- `src/gyo/web/app.js` — Rewrite for radial layout, interactions, animations

## API Changes Required

### New endpoint: `GET /api/node/{prefix}/metrics`

Returns:
```json
{
  "prefix": [0, 42],
  "level": 2,
  "occupancy": 150,
  "mean_residual": 0.23,
  "residual_norm": 0.45,
  "purity": 0.87,
  "size_norm": 0.6,
  "label_distribution": {
    "T-shirt_top": 45,
    "Trouser": 30,
    "Pullover": 25
  }
}
```

### Existing endpoint changes: `GET /api/tree`

Add to each node:
- `residual_norm` (already exists)
- `size_norm` (already exists)
- `purity` (already exists)

No changes needed — data is already sufficient.

## Implementation Approach

**Evolution of current code:**
- Keep D3 v7 as dependency
- Rewrite `app.js` for radial layout (approx 150-200 lines)
- Rewrite `style.css` for dark theme (approx 150-200 lines)
- Minimal changes to `index.html` structure
- Add `metrics` endpoint to `server.py` (approx 20 lines)

## Testing

- Visual verification: dark mode renders correctly
- Radial tree displays nodes in concentric rings
- Tooltips appear on hover
- Click zooms to node
- Double-click expands/collapses
- Side panel shows metrics + thumbnails
- All existing tests still pass
- New test for `/api/node/{prefix}/metrics` endpoint

## Non-Goals (YAGNI)

- Mobile responsiveness (desktop-only for now)
- Animation speed controls
- Multiple color themes
- Keyboard navigation
- Export/print functionality