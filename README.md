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
incompatible with the task runtime. The wrapper therefore implements only its
small image-preprocessing recipe and calls the pinned upstream Cellpose package
directly; SPArrOW's SpatialData orchestration, transcript allocation, QC, and
clustering stages are intentionally omitted because they do not define the
submitted boundaries. A reproducible compatibility wheel changes only
Cellpose's dependency metadata so its unchanged code can use the task runtime's
Python-3.13-compatible NumPy version.

The exact Cellpose 3 `cyto` weights are downloaded before candidate freezing
from the upstream model endpoint and accepted only when their size and SHA-256
match the committed identity. Cellpose notes that this model was trained in
part on CC-BY-NC data.

The default preset uses SPArrOW's committed Vizgen configuration rather than
the diameter sweep performed on this benchmark's validation field. That older
comparison layer used a different z-plane and `cyto3`, so it is not the output
of this reference.
