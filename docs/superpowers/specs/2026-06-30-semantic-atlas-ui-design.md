# Gyo Semantic Atlas — UI Design

**Status:** Approved for planning
**Date:** 2026-06-30

## 1. Purpose

Replace the current icicle interface with an intuitive visual explorer for evaluating embedding quality and residual quantization. The primary task is to determine quickly whether each group is visually coherent. The hierarchy remains essential: it must explain how residual tokens refine a parent group and whether those refinements are semantically meaningful.

## 2. Design principles

- Images are the primary evidence and must remain large enough to judge visually.
- Hierarchy provides orientation and explains residual refinement; it must not consume most of the canvas.
- Spatial proximity must have a defined mathematical meaning.
- Important actions must have visible controls. Gestures may accelerate actions but cannot be the only way to discover them.
- Projected distances are approximate; exact high-dimensional distances remain available.
- The interface must distinguish measured properties from visual approximations.

## 3. Main layout

The interface uses two persistent regions:

- A hierarchical semantic map occupies about 70% of the viewport width.
- A group inspector occupies about 30% and updates when a territory is selected.

The map begins at the root and displays its descendants as nested territories. Containment and connecting paths encode exact RQ parent-child relationships. Positions among siblings approximate distances between their reconstructed semantic vectors. Proximity is therefore interpreted among siblings, not as an unrestricted global distance.

Each territory contains a small, legible sample of representative images. Territory area encodes occupancy. Residual is encoded by a restrained border treatment rather than a large fill, preserving image fidelity. Selection and hover highlight the full ancestor path.

## 4. Geometry

### 4.1 Semantic vectors

For a prefix `(c0, ..., cn)`, the backend computes:

- `reconstructed_vector`: the sum of codebook vectors from level 0 through level n;
- `token_vector`: the codebook vector introduced at level n;
- `parent_distance`: the distance from the reconstructed parent vector to the reconstructed child vector;
- pairwise distances between reconstructed vectors for sibling groups.

Numeric token IDs are categorical identifiers. Their integer differences are never treated as semantic distances.
All first-version distances use Euclidean distance, matching the quantizer's `_assign` operation. Introducing cosine distance would create a second semantic criterion and is outside this scope.

### 4.2 Hierarchical metric MDS

Sibling positions use metric multidimensional scaling (MDS) from their pairwise distances. The layout then applies parent containment and collision resolution. A deterministic seed and deterministic initialization make repeated results stable. Geometry is computed by the backend and persisted with the dataset rather than recomputed on every page load.

Each sibling projection exposes normalized stress. The UI displays a warning when post-constraint normalized stress exceeds `0.10`. This threshold is configurable but has one shared default across datasets so comparisons remain consistent.

Containment and collision handling can distort the raw MDS coordinates. Therefore:

- map proximity is labeled as projected and approximate;
- exact original-space distances appear in the inspector;
- hierarchy is always represented by containment and paths, independently of geometry;
- projection quality is evaluated after containment, not only before it.

PCA may be used as a deterministic initializer for metric MDS but is not the final layout. UMAP is excluded from the first version because its local-neighborhood emphasis and parameter sensitivity make the map harder to audit. It can be evaluated later as an explicitly exploratory mode.

## 5. Group inspector

Selecting a territory updates the persistent side panel with:

- large representative images nearest to the reconstructed group vector;
- an explicit outlier view containing the farthest member images;
- occupancy, mean residual, normalized residual and purity when labels exist;
- exact distance from the parent;
- current token vector identifier and its residual contribution;
- MDS stress and a projection-quality explanation;
- an action to compare the selected child with its parent.

Parent-child comparison switches or juxtaposes representative image samples and reports the displacement introduced by the current residual token. Missing label or purity data is omitted rather than represented as zero.

## 6. Interaction model

1. Initial load shows the root's first territories and representative samples.
2. Hover highlights a group, its siblings and the complete path to root.
3. Single click selects a group and updates the inspector.
4. Double click enters a branch, but every selected non-leaf group also exposes a visible **Enter group** action.
5. Breadcrumbs return to ancestors without losing orientation.
6. A level control compares sibling groups at the same RQ stage.
7. **Compare with parent** explains the residual refinement.
8. **Show outliers** replaces representative samples with distant member images.
9. Zoom and pan support dense maps, with visible reset controls.

Single click never collapses a branch. Collapse is not part of the first version. Keyboard navigation and accessible labels must cover group selection, entering a group, returning to an ancestor and switching inspector modes.

## 7. Visual language

The approved direction is **scientific instrument**:

- dark, neutral canvas and panels;
- restrained borders and low-saturation surfaces;
- imagery receives the highest contrast;
- semantic state uses a small accent palette;
- numeric data uses tabular figures;
- no decorative glow or saturated territory fills that compete with images;
- skeletons preserve layout while thumbnails load.

The map should feel exploratory but remain credible as an analytical instrument. Animation communicates spatial continuity during zoom and focus changes; it does not decorate routine state changes.

## 8. Data contracts

The existing `GET /api/tree`, `GET /api/node/{prefix}`, `GET /api/node/{prefix}/metrics` and `GET /thumb/{idx}` endpoints remain useful but are insufficient for geometry and representative sampling.

The backend must provide, either through enriched existing contracts or a focused geometry endpoint:

- persisted normalized 2D center and base radius for every prefix; the browser fits these values to the current viewport without changing their relative geometry;
- parent and sibling relationships;
- reconstructed and token-vector distance summaries;
- projection stress;
- representative item IDs ranked by distance to the reconstructed vector;
- outlier item IDs ranked in the opposite direction.

The browser does not need raw high-dimensional vectors. Exact vectors remain server-side to limit payload size and keep the numerical implementation in one place.

## 9. Component boundaries

- **Geometry service:** loads codebooks, reconstructs prefix vectors, computes distances, runs deterministic metric MDS and persists geometry.
- **Tree API:** exposes hierarchy, geometry references and aggregate node metrics.
- **Sampling service:** ranks member embeddings into representative and outlier samples.
- **Map model:** normalizes API responses into immutable nodes and selection state.
- **Map renderer:** renders territories, hierarchy paths, samples and semantic states.
- **Viewport controller:** owns zoom, pan, focus transitions and fit/reset behavior.
- **Inspector:** renders samples, metrics and parent-child comparison.
- **Interaction controller:** centralizes pointer and keyboard actions so renderer behavior remains testable.

Each boundary must have a small explicit interface. Geometry calculation and layout normalization remain DOM-free.

## 10. Loading, failure and scale

- Structure renders before thumbnails; image slots use stable skeletons.
- Only visible territories and inspector samples load thumbnails.
- Geometry and metrics responses are cached per dataset version.
- Dense levels aggregate territories below a legibility threshold and reveal them after zoom or branch entry.
- Sparse groups enlarge available images without duplicating them.
- A failed thumbnail affects only its slot and provides a retry state.
- API failure preserves the last successful map when possible and exposes a retry action.
- High projection stress produces a warning without hiding the data.

## 11. Validation

### Backend tests

- Reconstructed vectors equal the sum of their path codewords.
- Parent distances and sibling distance matrices are correct.
- Representative and outlier rankings use the selected original-space metric correctly.
- Metric MDS is deterministic for the same dataset and configuration.
- Persisted geometry is invalidated when codebooks or dataset identity change.

### Layout tests

- Children remain contained by their parent territory.
- Visible territories do not overlap beyond an explicit tolerance.
- Post-constraint stress is measured and reported.
- Stable input produces stable coordinates.
- Dense groups aggregate only below the defined visual legibility threshold.

### Interaction tests

- Selection updates the inspector.
- Enter, breadcrumb return, zoom reset and level selection preserve orientation.
- Parent comparison and outlier modes use the correct samples.
- All core actions work with keyboard and pointer input.

### Visual and usability checks

- Representative images remain judgeable at supported viewport sizes.
- Loading and failure states do not shift map geometry.
- A user can identify whether a group is visually coherent within a few seconds.
- A user can explain the selected group's location in the hierarchy and what the current residual token added.

## 12. Scope

### Included

- Hierarchical semantic map using deterministic metric MDS among siblings.
- Persistent group inspector.
- Representative, outlier and parent-child comparison workflows.
- Scientific-instrument visual system.
- Responsive desktop and tablet layout with accessible core interactions.

### Excluded from the first version

- UMAP, t-SNE, PaCMAP or TriMap view modes.
- Comparing multiple embedders in one screen.
- Editing or retraining the quantizer.
- Free-form user annotations.
- Persisting arbitrary personal workspaces.
