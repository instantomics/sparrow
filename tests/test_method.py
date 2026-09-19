from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy import sparse
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

CANDIDATE_SOURCE = Path(__file__).parents[1] / "candidate/src"
sys.path.insert(0, str(CANDIDATE_SOURCE))


@dataclass(frozen=True)
class PolygonInstance:
    instance_id: str
    vertices: np.ndarray
    parent_cell_id: str | None = None
    interior_rings: tuple[np.ndarray, ...] = ()
    type_probabilities: dict[str, float] | None = None


@dataclass(frozen=True)
class SegmentationPrediction:
    cells: tuple[PolygonInstance, ...]
    nuclei: tuple[PolygonInstance, ...]


segmentation = ModuleType("segmentation")
segmentation.PolygonInstance = PolygonInstance
segmentation.SegmentationPrediction = SegmentationPrediction
segmentation.__path__ = []
dataset_io = ModuleType("segmentation.dataset_io")
dataset_io._load_ngff_image = None
schema = ModuleType("segmentation.schema")
schema.MAX_POLYGON_VERTICES = 1_024
cv2 = ModuleType("cv2")
torch = ModuleType("torch")
cellpose = ModuleType("cellpose")
cellpose.models = ModuleType("cellpose.models")

# The reference author environment does not install the task. Keep only the
# task boundary and expensive image model as scoped doubles; numerical typing
# and geometry use their real libraries.
with patch.dict(
    sys.modules,
    {
        "segmentation": segmentation,
        "segmentation.dataset_io": dataset_io,
        "segmentation.schema": schema,
        "cv2": cv2,
        "torch": torch,
        "cellpose": cellpose,
        "cellpose.models": cellpose.models,
    },
):
    from mymodel import method


def test_label_polygon_uses_ngff_transform_and_field_clip():
    labels = np.zeros((4, 5), dtype=np.int32)
    labels[1:3, 1:4] = 7

    cells = method._labels_to_cells(
        labels,
        origin=(10.0, 20.0),
        pixel_size=(2.0, 3.0),
        field_bounds=(13.0, 22.0, 17.0, 28.0),
    )

    assert len(cells) == 1
    assert cells[0].instance_id == "sparrow-cell-7"
    assert np.allclose(cells[0].vertices.min(axis=0), [13.0, 23.0])
    assert np.allclose(cells[0].vertices.max(axis=0), [17.0, 28.0])


def test_hole_fragments_preserve_mask_and_enclosed_cell():
    labels = np.ones((9, 9), dtype=np.int32)
    labels[2:7, 2:7] = 0
    labels[3:6, 3:6] = 2
    cells = method._labels_to_cells(
        labels, origin=(0, 0), pixel_size=(1, 1), field_bounds=(0, 0, 9, 9)
    )
    ring = [
        Polygon(cell.vertices) for cell in cells if cell.instance_id.startswith("sparrow-cell-1")
    ]
    center = next(Polygon(cell.vertices) for cell in cells if cell.instance_id == "sparrow-cell-2")
    assert len(ring) > 1
    assert all(not cell.interior_rings for cell in cells)
    assert unary_union(ring).equals(box(0, 0, 9, 9).difference(box(2, 2, 7, 7)))
    assert center.equals(box(3, 3, 6, 6))
    assert sum(part.area for part in ring) == unary_union(ring).area
    assert all(part.intersection(center).area == 0 for part in ring)


def test_posterior_types_final_geometry_from_visible_field(monkeypatch, tmp_path):
    model_path = tmp_path / "model"
    model_path.touch()
    monkeypatch.setattr(method, "_MODEL_PATH", model_path)
    image = SimpleNamespace(
        image=np.ones((2, 6), dtype=np.uint16), origin_um=(10, 20), pixel_size_um=(2, 3)
    )
    masks = np.array([[7, 7, 0, 9, 9, 0], [7, 7, 0, 9, 9, 0]])
    monkeypatch.setattr(method, "_load_ngff_image", lambda *a, **k: image)
    monkeypatch.setattr(method, "_preprocess", lambda values, config: values)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None, raising=False)
    monkeypatch.setattr(torch, "device", lambda name: name, raising=False)
    monkeypatch.setattr(
        cellpose.models,
        "CellposeModel",
        lambda **kwargs: SimpleNamespace(eval=lambda **kw: ([masks],)),
        raising=False,
    )
    # Reference projection order differs from the spatial gene axis. A boundary
    # transcript and an unmatched feature must not contribute to the first cell.
    transcripts = SimpleNamespace(
        gene_ids=("B", "unmatched", "A"),
        gene_index=np.array([2, 2, 0, 1, 0]),
        coordinates=np.array([[13, 22], [13, 23], [12, 22], [13, 24], [17, 22]]),
    )
    calls = []

    def matched(genes):
        calls.append(tuple(genes))
        return SimpleNamespace(gene_ids=("A", "B"), counts=sparse.csr_matrix([[9, 1], [1, 9]]))

    reference = SimpleNamespace(cell_type_labels=("alpha", "beta"), matched_expression=matched)
    field = SimpleNamespace(
        field_handle="visible-field",
        information_condition="labeled_reference",
        image_channels={"cytoplasm": tmp_path / "image"},
        nuclear_image=True,
        load_nuclear_image=lambda: image,
        field_bounds=(12, 20, 22, 26),
        load_reference=lambda: reference,
        load_transcripts=lambda: transcripts,
    )
    config = method.SparrowConfig()
    monkeypatch.setattr(method, "load_config", lambda path: config)
    submissions = []
    failures = []
    outputs = SimpleNamespace(
        submit=lambda field, prediction: submissions.append(prediction),
        fail=lambda field, error: failures.append(error),
    )
    method.posterior_method(
        SimpleNamespace(
            data=SimpleNamespace(list_fields=lambda: [field]), parameterization_path="unused"
        ),
        outputs,
    )
    assert not failures
    (prediction,) = submissions
    first, second = prediction.cells
    assert prediction.nuclei == ()
    assert first.type_probabilities["alpha"] > 0.9
    assert second.type_probabilities["beta"] > 0.8
    assert calls == [transcripts.gene_ids]
    assert first.vertices[:, 0].min() == 12
    # Removing all RNA changes probabilities but cannot move or remove cells.
    empty = SimpleNamespace(
        gene_ids=transcripts.gene_ids,
        gene_index=np.array([], dtype=int),
        coordinates=np.empty((0, 2)),
    )
    typed_empty = method.type_cells(
        prediction.cells, empty, reference, concentration=20, smoothing=0.05
    )
    for original, cell in zip(prediction.cells, typed_empty, strict=True):
        assert cell.instance_id == original.instance_id
        assert cell.vertices is original.vertices
        assert cell.type_probabilities == {"alpha": 0.5, "beta": 0.5}
