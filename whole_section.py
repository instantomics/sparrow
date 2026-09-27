from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import logging
import math
import os
import shutil
import sys
import time
import tomllib
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cellpose
import geopandas as gpd
import numpy as np
import tifffile
import torch
from cellpose import models
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box

LOGGER = logging.getLogger("sparrow-whole-section")
MAX_POLYGON_VERTICES = 1_024


@dataclass(frozen=True)
class Tile:
    row: int
    column: int
    core_x0: int
    core_y0: int
    core_x1: int
    core_y1: int
    window_x0: int
    window_y0: int
    window_x1: int
    window_y1: int
    core_rows: int
    core_columns: int
    image_width: int
    image_height: int


@dataclass(frozen=True)
class _PolygonInstance:
    """The candidate's task transport type, reproduced only at its import boundary."""

    instance_id: str
    vertices: np.ndarray
    parent_cell_id: str | None = None
    interior_rings: tuple[np.ndarray, ...] = ()
    type_probabilities: dict[str, float] | None = None


@dataclass(frozen=True)
class _SegmentationPrediction:
    cells: tuple[_PolygonInstance, ...]
    nuclei: tuple[_PolygonInstance, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download_verified(url: str, destination: Path, size: int, sha256: str) -> None:
    temporary = destination.with_suffix(".download")
    digest = hashlib.sha256()
    downloaded = 0
    request = urllib.request.Request(url, headers={"User-Agent": "iomix-sparrow-reference"})
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as stream,
        ):
            while block := response.read(1024 * 1024):
                downloaded += len(block)
                if downloaded > size:
                    raise ValueError("Cellpose model download exceeds its declared size")
                digest.update(block)
                stream.write(block)
        if downloaded != size or digest.hexdigest() != sha256:
            raise ValueError("Cellpose model download does not match its declared identity")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--section-id", required=True)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--reference-commit", required=True)
    parser.add_argument("--environment-receipt", required=True, type=Path)
    parser.add_argument("--candidate-source", required=True, type=Path)
    return parser.parse_args()


def _select_section(config: dict, section_id: str) -> dict:
    matches = [section for section in config["sections"] if section["section_id"] == section_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate section identity: {section_id}")
    return matches[0]


def _validate_input(path: Path, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected_size or _sha256(path) != expected_sha256:
        raise ValueError(f"input does not match its declared identity: {path}")


def _sparrow_package(receipt: dict) -> dict:
    matches = [item for item in receipt["packages"] if item["project_id"] == "reference:sparrow"]
    if len(matches) != 1:
        raise ValueError("environment receipt does not identify the SPArrOW reference")
    return matches[0]


def _validate_realized_environment(bootstrap: dict, realized: dict) -> dict:
    for key in ("target", "profile"):
        if realized.get(key) != bootstrap.get(key):
            raise ValueError(f"realized environment changed {key}")
    expected = _sparrow_package(bootstrap)
    actual = _sparrow_package(realized)
    for key in ("commit", "lock_digest", "pyproject_digest"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"realized SPArrOW environment changed {key}")
    if expected.get("dirty") or actual.get("dirty"):
        raise ValueError("whole-section execution requires a clean committed reference")
    for key in ("abi", "version", "executable"):
        if realized["python"].get(key) != bootstrap["python"].get(key):
            raise ValueError(f"realized environment changed Python {key}")
    return actual


def _validate_candidate_source(root: Path, declared: dict) -> dict[str, str]:
    expected = {
        "iomix_candidate.json": declared["manifest_sha256"],
        "mymodel/__init__.py": declared["init_sha256"],
        "mymodel/method.py": declared["method_sha256"],
        "mymodel/cell_typing.py": declared["cell_typing_sha256"],
        "mymodel/native_annotation.py": declared["native_annotation_sha256"],
    }
    observed = {}
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[relative] = _sha256(path)
        if observed[relative] != digest:
            raise ValueError(f"staged candidate source changed: {relative}")
    return observed


def _install_candidate_contract_shim() -> None:
    if "segmentation" in sys.modules:
        return
    segmentation = types.ModuleType("segmentation")
    segmentation.__path__ = []
    segmentation.PolygonInstance = _PolygonInstance
    segmentation.SegmentationPrediction = _SegmentationPrediction
    dataset_io = types.ModuleType("segmentation.dataset_io")

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("the whole-section adapter does not use task NGFF loading")

    dataset_io._load_ngff_image = unavailable
    schema = types.ModuleType("segmentation.schema")
    schema.MAX_POLYGON_VERTICES = MAX_POLYGON_VERTICES
    sys.modules.update(
        {
            "segmentation": segmentation,
            "segmentation.dataset_io": dataset_io,
            "segmentation.schema": schema,
        }
    )


def _load_candidate_method(source_root: Path, declared: dict):
    _validate_candidate_source(source_root, declared)
    _install_candidate_contract_shim()
    source_parent = str(source_root.resolve())
    if source_parent not in sys.path:
        sys.path.insert(0, source_parent)
    method = importlib.import_module("mymodel.method")
    loaded_path = Path(method.__file__).resolve()
    if loaded_path != (source_root / "mymodel/method.py").resolve():
        raise RuntimeError(f"loaded candidate method from an unexpected path: {loaded_path}")
    return method


def _axis_aligned_pixel_geometry(micron_to_pixel: np.ndarray) -> tuple[float, float]:
    if micron_to_pixel.shape != (3, 3):
        raise ValueError("affine must be a homogeneous 3x3 micron-to-pixel transform")
    if not np.allclose(micron_to_pixel[2], [0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError("affine must be a homogeneous 3x3 micron-to-pixel transform")
    if not np.allclose(micron_to_pixel[[0, 1], [1, 0]], 0, rtol=0, atol=1e-12):
        raise ValueError("whole-section candidate windows require an axis-aligned affine")
    scale_x = float(micron_to_pixel[0, 0])
    scale_y = float(micron_to_pixel[1, 1])
    if not math.isfinite(scale_x) or not math.isfinite(scale_y) or min(scale_x, scale_y) <= 0:
        raise ValueError("affine pixel scales must be positive and finite")
    return scale_x, scale_y


def _validate_adaptation(adaptation: dict) -> None:
    for name in ("candidate_window_um", "owned_core_um", "halo_um"):
        value = adaptation[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    expected_window = adaptation["owned_core_um"] + 2 * adaptation["halo_um"]
    if not math.isclose(adaptation["candidate_window_um"], expected_window):
        raise ValueError("candidate window must equal the owned core plus both halos")


def _tile_plan(
    image_shape: tuple[int, int],
    *,
    scale_x: float,
    scale_y: float,
    core_um: float,
    halo_um: float,
) -> tuple[list[Tile], dict[str, int]]:
    height, width = image_shape
    core_x = max(1, round(core_um * scale_x))
    core_y = max(1, round(core_um * scale_y))
    # Ceil guarantees at least the declared physical halo after pixelization.
    halo_x = max(1, math.ceil(halo_um * scale_x))
    halo_y = max(1, math.ceil(halo_um * scale_y))
    window_x = min(width, core_x + 2 * halo_x)
    window_y = min(height, core_y + 2 * halo_y)
    columns = math.ceil(width / core_x)
    rows = math.ceil(height / core_y)
    tiles = []
    for row in range(rows):
        core_y0 = row * core_y
        core_y1 = min(height, core_y0 + core_y)
        window_y0 = min(max(0, core_y0 - halo_y), height - window_y)
        for column in range(columns):
            core_x0 = column * core_x
            core_x1 = min(width, core_x0 + core_x)
            window_x0 = min(max(0, core_x0 - halo_x), width - window_x)
            tiles.append(
                Tile(
                    row=row,
                    column=column,
                    core_x0=core_x0,
                    core_y0=core_y0,
                    core_x1=core_x1,
                    core_y1=core_y1,
                    window_x0=window_x0,
                    window_y0=window_y0,
                    window_x1=window_x0 + window_x,
                    window_y1=window_y0 + window_y,
                    core_rows=rows,
                    core_columns=columns,
                    image_width=width,
                    image_height=height,
                )
            )
    return tiles, {
        "core_pixels_x": core_x,
        "core_pixels_y": core_y,
        "halo_pixels_x": halo_x,
        "halo_pixels_y": halo_y,
    }


def _tile_coordinates(tile: Tile, micron_to_pixel: np.ndarray) -> tuple[tuple, tuple, tuple]:
    scale_x, scale_y = _axis_aligned_pixel_geometry(micron_to_pixel)
    origin_x = (tile.window_x0 - micron_to_pixel[0, 2]) / scale_x
    origin_y = (tile.window_y0 - micron_to_pixel[1, 2]) / scale_y
    pixel_size = (1.0 / scale_x, 1.0 / scale_y)
    bounds = (
        origin_x,
        origin_y,
        origin_x + (tile.window_x1 - tile.window_x0) * pixel_size[0],
        origin_y + (tile.window_y1 - tile.window_y0) * pixel_size[1],
    )
    return (origin_x, origin_y), pixel_size, bounds


def _core_geometry(tile: Tile, micron_to_pixel: np.ndarray):
    scale_x, scale_y = _axis_aligned_pixel_geometry(micron_to_pixel)
    return box(
        (tile.core_x0 - micron_to_pixel[0, 2]) / scale_x,
        (tile.core_y0 - micron_to_pixel[1, 2]) / scale_y,
        (tile.core_x1 - micron_to_pixel[0, 2]) / scale_x,
        (tile.core_y1 - micron_to_pixel[1, 2]) / scale_y,
    )


def _polygon_parts(geometry) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        return [part for child in geometry.geoms for part in _polygon_parts(child)]
    return []


def _top_type(probabilities: dict[str, float] | None) -> str | None:
    if not probabilities:
        return None
    return min(probabilities, key=lambda name: (-probabilities[name], name))


def _software_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> None:
    args = _parse_args()
    started = time.monotonic()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    section = _select_section(config, args.section_id)
    parameters = config["parameters"]
    adaptation = config["adaptation"]
    method_config = config["method"]
    model_config = config["model"]
    provenance = config["provenance"]
    if method_config["stack_order"] != ["DAPI", "PolyT"]:
        raise ValueError("current candidate stack order must be DAPI then PolyT")
    if method_config["cellpose_channels"] != [2, 1]:
        raise ValueError("current candidate Cellpose channels must be [2, 1]")

    workspace_root = args.workspace_root.resolve()
    input_root = workspace_root / section["input_root"]
    dapi_path = input_root / "dapi.tif"
    polyt_path = input_root / "polyt.tif"
    affine_path = input_root / "affine.csv"
    output = workspace_root / section["output"]
    receipt_path = workspace_root / section["receipt"]
    scratch_root = os.environ.get("SCRATCHDIR")
    if not scratch_root:
        raise RuntimeError("SCRATCHDIR is required for allocation-local execution")
    work_dir = (Path(scratch_root) / f"sparrow-current-{args.section_id}").resolve()
    if output.is_relative_to(work_dir) or receipt_path.is_relative_to(work_dir):
        raise ValueError("durable outputs must be outside the disposable work directory")
    if work_dir.exists() or output.exists() or receipt_path.exists():
        raise FileExistsError("whole-section execution requires fresh output paths")
    if not args.environment_receipt.is_file():
        raise FileNotFoundError(args.environment_receipt)
    realized_receipt_value = os.environ.get("IOM_ACTIVE_SHELL_RECEIPT")
    if not realized_receipt_value:
        raise RuntimeError("IOM_ACTIVE_SHELL_RECEIPT is unavailable")
    realized_receipt_path = Path(realized_receipt_value).resolve()
    bootstrap_receipt = json.loads(args.environment_receipt.read_text(encoding="utf-8"))
    realized_receipt = json.loads(realized_receipt_path.read_text(encoding="utf-8"))
    realized_package = _validate_realized_environment(bootstrap_receipt, realized_receipt)
    if args.reference_commit != realized_package["commit"]:
        raise ValueError("requested reference commit does not match the realized environment")
    if not torch.cuda.is_available():
        raise RuntimeError("the whole-section adaptation requires a CUDA GPU")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name, path in (("dapi", dapi_path), ("polyt", polyt_path), ("affine", affine_path)):
        LOGGER.info("validating %s input", name)
        _validate_input(path, section[f"{name}_size"], section[f"{name}_sha256"])

    candidate_method = _load_candidate_method(args.candidate_source.resolve(), config["candidate_source"])
    candidate_config = candidate_method.SparrowConfig(**parameters)
    candidate_method._validate_config(candidate_config)
    if candidate_config != candidate_method.SparrowConfig():
        raise ValueError("whole-section parameters differ from the bound candidate preset")

    dapi = tifffile.memmap(dapi_path)
    polyt = tifffile.memmap(polyt_path)
    expected_shape = tuple(section["image_shape"])
    if dapi.shape != expected_shape or polyt.shape != expected_shape:
        raise ValueError("image shape does not match the declared section")
    if dapi.dtype != polyt.dtype or dapi.ndim != 2 or dapi.dtype != np.dtype("uint16"):
        raise TypeError("reviewed inputs must be matching two-dimensional uint16 images")
    micron_to_pixel = np.loadtxt(affine_path)
    scale_x, scale_y = _axis_aligned_pixel_geometry(micron_to_pixel)
    _validate_adaptation(adaptation)
    tiles, pixelization = _tile_plan(
        expected_shape,
        scale_x=scale_x,
        scale_y=scale_y,
        core_um=adaptation["owned_core_um"],
        halo_um=adaptation["halo_um"],
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True)
    output_temporary = output.with_suffix(output.suffix + ".partial")
    receipt_temporary = receipt_path.with_suffix(receipt_path.suffix + ".partial")
    output_temporary.unlink(missing_ok=True)
    receipt_temporary.unlink(missing_ok=True)
    published_output = False
    try:
        model_path = work_dir / model_config["file_name"]
        LOGGER.info("staging pinned Cellpose cytotorch_0 model")
        _download_verified(
            model_config["url"],
            model_path,
            model_config["size_bytes"],
            model_config["sha256"],
        )
        torch.set_num_threads(candidate_config.torch_threads)
        torch.cuda.set_device(0)
        model = models.CellposeModel(
            gpu=True,
            pretrained_model=str(model_path),
            device=torch.device(adaptation["device"]),
        )

        records = []
        ownership_eliminated = 0
        seam_clipped = 0
        LOGGER.info("running %d haloed candidate windows", len(tiles))
        for tile_number, tile in enumerate(tiles, start=1):
            window = np.stack(
                (
                    candidate_method._preprocess(
                        dapi[tile.window_y0 : tile.window_y1, tile.window_x0 : tile.window_x1],
                        candidate_config,
                    ),
                    candidate_method._preprocess(
                        polyt[tile.window_y0 : tile.window_y1, tile.window_x0 : tile.window_x1],
                        candidate_config,
                    ),
                ),
                axis=-1,
            )
            masks = model.eval(
                x=[window],
                batch_size=candidate_config.batch_size,
                channels=method_config["cellpose_channels"],
                channel_axis=2,
                normalize=True,
                diameter=candidate_config.diameter,
                flow_threshold=candidate_config.flow_threshold,
                cellprob_threshold=candidate_config.cellprob_threshold,
                do_3D=False,
                min_size=candidate_config.min_size,
                augment=False,
                tile_overlap=0.1,
                compute_masks=True,
            )[0][0]
            origin, pixel_size, field_bounds = _tile_coordinates(tile, micron_to_pixel)
            cells = candidate_method._labels_to_cells(
                np.asarray(masks),
                origin=origin,
                pixel_size=pixel_size,
                field_bounds=field_bounds,
            )
            core = _core_geometry(tile, micron_to_pixel)
            for cell in cells:
                geometry = Polygon(cell.vertices, cell.interior_rings)
                pieces = [part for part in _polygon_parts(geometry.intersection(core)) if part.area > 0]
                if not pieces:
                    ownership_eliminated += 1
                    continue
                if not core.covers(geometry):
                    seam_clipped += 1
                probabilities = cell.type_probabilities
                for piece_index, piece in enumerate(pieces):
                    record = {
                        "cell_id": (
                            f"{args.section_id}-sparrow-r{tile.row:04d}-c{tile.column:04d}-"
                            f"{cell.instance_id}-p{piece_index:03d}"
                        ),
                        "geometry": piece,
                    }
                    if probabilities is not None:
                        record["type_probabilities"] = json.dumps(probabilities, sort_keys=True)
                        record["top_type"] = _top_type(probabilities)
                    records.append(record)
            if tile_number == 1 or tile_number % 25 == 0 or tile_number == len(tiles):
                LOGGER.info(
                    "completed window %d/%d; retained=%d ownership_eliminated=%d",
                    tile_number,
                    len(tiles),
                    len(records),
                    ownership_eliminated,
                )

        if records:
            cells = gpd.GeoDataFrame(records, geometry="geometry")
        else:
            cells = gpd.GeoDataFrame(
                {"cell_id": []}, geometry=gpd.GeoSeries([], name="geometry")
            )
        cells.to_parquet(output_temporary, index=False)
        output_sha256 = _sha256(output_temporary)
        actual_window_um = {
            "owned_core_x": pixelization["core_pixels_x"] / scale_x,
            "owned_core_y": pixelization["core_pixels_y"] / scale_y,
            "halo_x": pixelization["halo_pixels_x"] / scale_x,
            "halo_y": pixelization["halo_pixels_y"] / scale_y,
            "interior_window_x": (
                pixelization["core_pixels_x"] + 2 * pixelization["halo_pixels_x"]
            )
            / scale_x,
            "interior_window_y": (
                pixelization["core_pixels_y"] + 2 * pixelization["halo_pixels_y"]
            )
            / scale_y,
        }
        receipt = {
            "schema_version": 1,
            "reference_id": "sparrow",
            "reference_commit": args.reference_commit,
            "section_id": args.section_id,
            "scope": "whole_section",
            "coordinate_system": "global_um",
            "segmentation_id": method_config["segmentation_id"],
            "evaluation_id": provenance["evaluation_id"],
            "run_id": provenance["run_id"],
            "candidate_id": provenance["candidate_id"],
            "candidate_revision": provenance["candidate_revision"],
            "architecture_id": provenance["architecture_id"],
            "parameterization_id": provenance["parameterization_id"],
            "task_release_id": provenance["task_release_id"],
            "task_commit": provenance["task_commit"],
            "canonical_score": provenance["score"],
            "upstream_repository": method_config["upstream_repository"],
            "upstream_commit": method_config["upstream_commit"],
            "image_plane": method_config["image_plane"],
            "stack_order": method_config["stack_order"],
            "cellpose_channels": method_config["cellpose_channels"],
            "parameters": parameters,
            "adaptation": {
                **adaptation,
                "pixelization": pixelization,
                "actual_um": actual_window_um,
                "tile_count": len(tiles),
                "ownership_eliminated_cell_count": ownership_eliminated,
                "seam_clipped_cell_count": seam_clipped,
                "edge_windows_shifted_inward": True,
                "cross_tile_overlap_proof": "positive-area-disjoint_core_partition",
                "typing_status": "not_run_no_labeled_reference_input",
            },
            "inputs": {
                "affine_sha256": section["affine_sha256"],
                "dapi_sha256": section["dapi_sha256"],
                "polyt_sha256": section["polyt_sha256"],
            },
            "candidate_source_sha256": _validate_candidate_source(
                args.candidate_source.resolve(), config["candidate_source"]
            ),
            "cellpose_model_sha256": model_config["sha256"],
            "bootstrap_environment_receipt_sha256": _sha256(args.environment_receipt),
            "realized_environment_receipt_sha256": _sha256(realized_receipt_path),
            "entrypoint_sha256": _sha256(Path(__file__)),
            "config_sha256": _sha256(args.config),
            "software": {
                "cellpose": cellpose.version,
                "geopandas": _software_version("geopandas"),
                "numpy": np.__version__,
                "opencv": _software_version("opencv-python-headless"),
                "scipy": _software_version("scipy"),
                "torch": torch.__version__,
            },
            "resources": config["resources"],
            "n_cells": len(cells),
            "elapsed_seconds": time.monotonic() - started,
            "output_sha256": output_sha256,
        }
        receipt_temporary.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(output_temporary, output)
        published_output = True
        os.replace(receipt_temporary, receipt_path)
        LOGGER.info("published %d cells to %s", len(cells), output)
    finally:
        output_temporary.unlink(missing_ok=True)
        receipt_temporary.unlink(missing_ok=True)
        if published_output and not receipt_path.exists():
            output.unlink(missing_ok=True)
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
