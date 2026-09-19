# SPArrOW

This reference adapts the image-segmentation and marker-based annotation stages of
[SPArrOW](https://github.com/saeyslab/napari-sparrow) at commit
`fcdb27cabe51cbb8b3f0f6d2b2a58530e9fbc43f` to the Iomix segmentation
contract. It follows SPArrOW's Vizgen recipe: min-max background removal and
CLAHE are applied to the registered cytoplasm and nuclear channels, then
Cellpose `cyto` segments whole cells using cytoplasm as the primary channel and
nuclear signal as the secondary channel.

The wrapper uses the task's middle-plane images and treats a registered
cytoplasm channel as Poly-T-compatible. A field without aligned cytoplasm and
nuclear images fails rather than silently switching methods. It returns
whole-cell boundaries with reference-guided type probabilities when labels are
supplied, and no nuclei. The task owns the [candidate contract](../../tasks/segmentation/contracts/contract.md).

SPArrOW itself currently requires Python 3.11 and NumPy below 2, which is
incompatible with the task runtime. The benchmark candidate therefore
implements only its small image-preprocessing recipe and calls the pinned
upstream Cellpose package directly; SPArrOW's SpatialData orchestration,
QC and clustering stages are omitted. For typing, the candidate executes the
unchanged computational routines behind SPArrOW's iterative marker annotation,
without its SpatialData or plotting wrappers. The upstream source and license
are staged from the pinned revision before freezing. Marker discovery and
probability calibration are reference-owned adaptations described below. A
reproducible compatibility wheel changes only Cellpose's dependency metadata so
its unchanged code can use the task runtime's Python-3.13-compatible NumPy
version.

## Reference-guided typing

SPArrOW's image recipe remains the sole source of geometry. The wrapper counts
transcripts strictly inside each final submitted polygon, after clipping and
simplification, and uses only the supplied labeled single-cell reference for
typing. Shared features follow the task's name-matching and duplicate-feature
aggregation policy. Unmatched transcripts do not contribute typing evidence.

Markers are derived automatically from a deterministic discovery subset of the
visible labeled reference. Positive one-vs-rest enrichment is computed from
library-normalized expression, weighting the other types equally. A detection
requirement avoids markers supported by isolated reference reads. The strongest
eligible genes form a binary marker table supplied to SPArrOW; no manually
curated or evaluator-derived marker list is used. Library normalization,
log transformation and gene scaling use the discovery reference.

SPArrOW iteratively scores marker expression and recenters against its inferred
types. Calibration reference cells are scored jointly with the field's cells,
without their labels being passed to SPArrOW. The native number-of-markers
normalization is retained; hard score rejection is disabled so negative scores
remain available for probabilistic comparison. Opaque internal type IDs preserve
the exact external labels despite upstream name sanitization.

A disjoint reference subset supplies calibration labels. These cells are thinned
without replacement to cover sparse transcript depths, and a scalar temperature
is fitted with type-balanced log loss. Probabilities are shrunk toward the
equal-type prior according to observed marker depth; this same shrinkage is
included during calibration. Type priors are equal because reference sampling
frequencies need not describe tissue composition. The [preset](reference.toml)
fixes scientific settings and the stochastic seed, without evaluator feedback.

Empty cells and cells with no selected-marker RNA return the prior. Types with
too few reference cells for independent calibration, or no distinguishing panel
markers, retain their prior mass rather than being silently dropped. If fewer
than two types are supported, or held-out scores do not improve on the prior,
typing remains uniform. No `UNKNOWN` mass is introduced for these data deficits:
they do not establish membership in an unrepresented type. Reference-only
calibration does not establish calibration across platform shifts, and native
iterative centering is transductive: it depends on the visible scoring cohort.

The transport cannot represent holes. Hole-bearing masks are deterministically
cut into hole-free fragments without filling background or enclosed cells; each
fragment is typed from its own transcripts. This representation can change
instance counts and introduces internal cut boundaries, but preserves the
SPArrOW footprint before the existing vertex-budget simplification.

The exact Cellpose 3 `cyto` weights are downloaded before candidate freezing
from the upstream model endpoint and accepted only when their size and SHA-256
match the committed identity. Cellpose notes that this model was trained in
part on CC-BY-NC data.

The default preset uses SPArrOW's committed Vizgen configuration rather than
the diameter sweep performed on this benchmark's validation field. That older
comparison layer used a different z-plane and `cyto3`, so it is not the output
of this reference.

The separately declared `wagner-reviewed-z4` whole-section integration
reproduces that reviewed comparison layer for downstream Wagner analyses. It
uses z4 imagery, channel-specific SPArrOW preprocessing, Cellpose `cyto3`, and
the selected diameter of 100 pixels; it is not an alternative benchmark preset.
