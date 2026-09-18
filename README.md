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
whole-cell boundaries only.

SPArrOW itself currently requires Python 3.11 and NumPy below 2, which is
incompatible with the task runtime. The benchmark candidate therefore
implements only its small image-preprocessing recipe and calls the pinned
upstream Cellpose package directly; SPArrOW's SpatialData orchestration,
transcript allocation, QC, and clustering stages are intentionally omitted from
that candidate because they do not define the submitted boundaries. A
reproducible compatibility wheel changes only Cellpose's dependency metadata so
its unchanged code can use the task runtime's Python-3.13-compatible NumPy
version.

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
