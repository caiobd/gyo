### Semantic Atlas scale constraints

The first Semantic Atlas version has an explicit analytical product limit of `MAX_ATLAS_SIBLINGS = 256` real children at any focus. Datasets above that per-focus cardinality remain available through `/api/tree`, but Atlas geometry returns HTTP 409 and recommends training with a smaller codebook or lower per-level cardinality.

The browser renders at most 96 real territories at once. When a level is denser, it retains a stable occupancy-ranked visible subset and represents the remainder as a synthetic aggregate; zoom can reveal more up to that cap. Visible layout stress is computed only from the real territories currently displayed, excluding the synthetic aggregate.
