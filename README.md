# SPArrOW

This reference adapts the image-segmentation stage of
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
QC and clustering stages are omitted. Transcript allocation and supervised typing
are wrapper adaptations described below, not upstream SPArrOW cell-type inference. A
reproducible compatibility wheel changes only Cellpose's dependency metadata so
its unchanged code can use the task runtime's Python-3.13-compatible NumPy
version.

## Reference-guided typing

SPArrOW's image recipe remains the sole source of geometry. The wrapper counts
transcripts strictly inside each final submitted polygon, after clipping and
simplification, and uses only the supplied labeled single-cell reference for
typing. Shared features follow the task's name-matching and duplicate-feature
aggregation policy. Unmatched transcripts do not contribute typing evidence.

Each type's expression profile is the mean of shared-panel-normalized reference
cells, giving equal influence to cells rather than sequencing depth. A
Dirichlet-multinomial observation model provides probabilities over the exact
visible labels. Its finite concentration allows biological overdispersion instead
of treating every transcript as independent evidence for a fixed profile; a small
uniform gene component avoids impossibility claims from reference sampling zeros.
The parameters are fixed in the [preset](reference.toml), not tuned to evaluation
outcomes. Type priors are equal: reference sampling frequencies need not describe
the tissue's composition.

Empty cells, zero shared genes, and a wholly RNA-empty shared reference return
the equal-type prior. A single shared gene cannot distinguish normalized profiles;
few genes or transcripts yield only the evidence this model supports, without a
hard typing threshold. A reference type with no shared-panel RNA retains a
neutral mean profile rather than being ruled out. No `UNKNOWN` mass is introduced
for sparsity or panel mismatch: neither establishes that a cell belongs to an
unrepresented type. This is closed-set model uncertainty, not calibrated
out-of-reference detection. Platform effects and within-type heterogeneity can
still miscalibrate probabilities.

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
