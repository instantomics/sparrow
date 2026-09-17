from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

CANDIDATE_SOURCE = Path(__file__).parents[1] / "candidate/src"
sys.path.insert(0, str(CANDIDATE_SOURCE))


@dataclass(frozen=True)
class PolygonInstance:
    instance_id: str
    vertices: np.ndarray
    parent_cell_id: str | None = None
    interior_rings: tuple[np.ndarray, ...] = ()


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
sys.modules["segmentation"] = segmentation
sys.modules["segmentation.dataset_io"] = dataset_io
sys.modules["segmentation.schema"] = schema

cv2 = ModuleType("cv2")
sys.modules["cv2"] = cv2
torch = ModuleType("torch")
sys.modules["torch"] = torch
cellpose = ModuleType("cellpose")
cellpose.models = ModuleType("cellpose.models")
sys.modules["cellpose"] = cellpose
sys.modules["cellpose.models"] = cellpose.models
scipy = ModuleType("scipy")
scipy.ndimage = ModuleType("scipy.ndimage")
sys.modules["scipy"] = scipy
sys.modules["scipy.ndimage"] = scipy.ndimage

from mymodel import method  # noqa: E402


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
