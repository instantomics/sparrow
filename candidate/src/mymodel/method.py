from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from affine import Affine
from cellpose import models
from rasterio.features import shapes
from scipy import ndimage
from segmentation import PolygonInstance, SegmentationPrediction
from segmentation.dataset_io import _load_ngff_image
from segmentation.schema import MAX_POLYGON_VERTICES
from shapely import make_valid
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon, box, shape
from shapely.ops import split

from .cell_typing import MarkerTypingConfig, type_cells

LOGGER = logging.getLogger(__name__)
_MODEL_PATH = Path(__file__).with_name("cytotorch_0")


@dataclass(frozen=True)
class SparrowConfig(MarkerTypingConfig):
    background_filter_size: int = 135
    batch_size: int = 8
    cellprob_threshold: float = -4.0
    clahe_clip: float = 18.5
    diameter: float = 100.0
    flow_threshold: float = 0.85
    min_size: int = 80
    torch_threads: int = 4


def load_config(path: str | Path) -> SparrowConfig:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise TypeError("candidate parameterization must be a JSON object")
    reference = document.get("reference")
    parameters = reference.get("parameters") if isinstance(reference, Mapping) else None
    if not isinstance(parameters, Mapping):
        raise TypeError("candidate parameterization must contain reference parameters")
    unknown = set(parameters).difference(SparrowConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown SPArrOW parameters: {sorted(unknown)!r}")
    config = SparrowConfig(**dict(parameters))
    _validate_config(config)
    return config


def _validate_config(config: SparrowConfig) -> None:
    for name in ("background_filter_size", "batch_size", "min_size", "torch_threads"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.background_filter_size % 2 != 1:
        raise ValueError("background_filter_size must be odd")
    if config.torch_threads > 4:
        raise ValueError("torch_threads must not exceed the declared CPU count")
    for name in (
        "clahe_clip",
        "diameter",
        "flow_threshold",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    threshold = config.cellprob_threshold
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("cellprob_threshold must be a number")
    if not math.isfinite(threshold):
        raise ValueError("cellprob_threshold must be finite")


def posterior_method(inputs, outputs) -> None:
    fields = inputs.data.list_fields()
    try:
        config = load_config(inputs.parameterization_path)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        for field in fields:
            outputs.fail(field, error)
        return

    for field in fields:
        try:
            outputs.submit(field, segment_field(field, config))
        except Exception as error:
            LOGGER.exception("sparrow_field_failed field_handle=%s", field.field_handle)
            outputs.fail(field, error)


def segment_field(field, config: SparrowConfig) -> SegmentationPrediction:
    if not _MODEL_PATH.is_file():
        raise FileNotFoundError("the pinned Cellpose cyto model was not staged")
    cytoplasm_path = field.image_channels.get("cytoplasm")
    if cytoplasm_path is None or field.nuclear_image is None:
        raise ValueError("SPArrOW requires registered cytoplasm and nuclear images")

    nuclear = field.load_nuclear_image()
    cytoplasm = _load_ngff_image(
        cytoplasm_path,
        channel_id="cytoplasm",
        role="cytoplasm",
    )
    _validate_aligned_images(nuclear, cytoplasm)
    image = np.stack(
        (
            _preprocess(nuclear.image, config),
            _preprocess(cytoplasm.image, config),
        ),
        axis=-1,
    )

    torch.set_num_threads(config.torch_threads)
    model = models.CellposeModel(
        gpu=False,
        pretrained_model=str(_MODEL_PATH),
        device=torch.device("cpu"),
    )
    masks = model.eval(
        x=[image],
        batch_size=config.batch_size,
        channels=[2, 1],
        channel_axis=2,
        normalize=True,
        diameter=config.diameter,
        flow_threshold=config.flow_threshold,
        cellprob_threshold=config.cellprob_threshold,
        do_3D=False,
        min_size=config.min_size,
        augment=False,
        tile_overlap=0.1,
        compute_masks=True,
    )[0][0]
    cells = _labels_to_cells(
        np.asarray(masks),
        origin=nuclear.origin_um,
        pixel_size=nuclear.pixel_size_um,
        field_bounds=field.field_bounds,
    )
    if field.information_condition == "labeled_reference":
        reference = field.load_reference()
        if reference is None:
            raise ValueError("labeled-reference field did not supply a reference")
        cells = type_cells(
            cells,
            field.load_transcripts(),
            reference,
            config=config,
        )
    return SegmentationPrediction(cells=tuple(cells), nuclei=())


def _validate_aligned_images(nuclear, cytoplasm) -> None:
    if np.asarray(nuclear.image).shape != np.asarray(cytoplasm.image).shape:
        raise ValueError("cytoplasm and nuclear image shapes differ")
    if not np.allclose(nuclear.origin_um, cytoplasm.origin_um, rtol=0, atol=1e-9):
        raise ValueError("cytoplasm and nuclear image origins differ")
    if not np.allclose(nuclear.pixel_size_um, cytoplasm.pixel_size_um, rtol=0, atol=1e-12):
        raise ValueError("cytoplasm and nuclear image pixel sizes differ")


def _preprocess(values: np.ndarray, config: SparrowConfig) -> np.ndarray:
    image = np.asarray(values)
    if image.ndim != 2 or image.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise TypeError("SPArrOW preprocessing requires a two-dimensional uint8 or uint16 image")
    opened = ndimage.maximum_filter(
        ndimage.minimum_filter(image, size=config.background_filter_size),
        size=config.background_filter_size,
    )
    background_removed = image - opened
    clahe = cv2.createCLAHE(clipLimit=config.clahe_clip, tileGridSize=(8, 8))
    return clahe.apply(background_removed)


def _labels_to_cells(
    labels: np.ndarray,
    *,
    origin: Sequence[float],
    pixel_size: Sequence[float],
    field_bounds: Sequence[float],
) -> list[PolygonInstance]:
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("Cellpose labels must be a two-dimensional integer array")
    origin_x, origin_y = (float(value) for value in origin)
    scale_x, scale_y = (float(value) for value in pixel_size)
    transform = Affine(scale_x, 0.0, origin_x, 0.0, scale_y, origin_y)
    bounds = box(*(float(value) for value in field_bounds))
    by_label: dict[int, list[Polygon]] = {}
    values = labels.astype(np.int32, copy=False)
    for encoded, label in shapes(values, mask=values > 0, connectivity=8, transform=transform):
        polygon = _largest_polygon(make_valid(shape(encoded)).intersection(bounds))
        if polygon is not None:
            by_label.setdefault(int(label), []).append(polygon)

    accepted: list[Polygon] = []
    result = []
    for label, polygons in sorted(by_label.items()):
        polygon = max(enumerate(polygons), key=lambda item: (item[1].area, -item[0]))[1]
        polygon = _simplify_to_limit(polygon, min(scale_x, scale_y))
        for previous in accepted:
            if polygon is None:
                break
            if polygon.intersects(previous):
                polygon = _largest_polygon(make_valid(polygon.difference(previous)))
        if polygon is None:
            continue
        polygon = _simplify_to_limit(polygon, min(scale_x, scale_y))
        if polygon is None:
            continue
        parts = _hole_free_parts(polygon)
        for index, part in enumerate(parts):
            part = _simplify_to_limit(part, min(scale_x, scale_y))
            if part is None:
                continue
            exterior = np.asarray(part.exterior.coords[:-1], dtype=np.float64)
            suffix = f"-part-{index + 1}" if len(parts) > 1 else ""
            accepted.append(part)
            result.append(PolygonInstance(f"sparrow-cell-{label}{suffix}", exterior))
    return result


def _hole_free_parts(polygon: Polygon) -> list[Polygon]:
    # The transport admits exterior rings only. Cut through holes rather than
    # filling them (which would assign background or another cell's transcripts).
    pending = [polygon]
    result = []
    while pending:
        part = pending.pop()
        if not part.interiors:
            result.append(part)
            continue
        y = Polygon(part.interiors[0]).representative_point().y
        left, _, right, _ = part.bounds
        cut = LineString([(left - 1, y), (right + 1, y)])
        pieces = list(split(part, cut).geoms)
        if sum(len(piece.interiors) for piece in pieces) >= len(part.interiors):
            raise ValueError("could not represent a SPArrOW polygon without interior rings")
        pending.extend(pieces)
    return sorted(result, key=lambda part: part.bounds)


def _largest_polygon(geometry) -> Polygon | None:
    if isinstance(geometry, Polygon):
        return geometry if not geometry.is_empty and geometry.area > 0 else None
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        polygons = [
            item for child in geometry.geoms if (item := _largest_polygon(child)) is not None
        ]
        if polygons:
            return max(enumerate(polygons), key=lambda item: (item[1].area, -item[0]))[1]
    return None


def _vertex_count(polygon: Polygon) -> int:
    return (
        len(polygon.exterior.coords) - 1 + sum(len(ring.coords) - 1 for ring in polygon.interiors)
    )


def _simplify_to_limit(polygon: Polygon, pixel_size: float) -> Polygon | None:
    if _vertex_count(polygon) <= MAX_POLYGON_VERTICES:
        return polygon
    tolerance = pixel_size / 4
    for _ in range(16):
        simplified = _largest_polygon(
            make_valid(polygon.simplify(tolerance, preserve_topology=True))
        )
        if simplified is not None and _vertex_count(simplified) <= MAX_POLYGON_VERTICES:
            return simplified
        tolerance *= 2
    return None
