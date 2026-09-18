from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import tomllib
import urllib.request
from pathlib import Path

import cellpose
import dask
import dask.array as da
import geopandas as gpd
import numpy as np
import sparrow as sp
import spatialdata as sd
import tifffile
import torch
from shapely.affinity import affine_transform
from spatialdata import SpatialData
from spatialdata.models import Image2DModel

LOGGER = logging.getLogger("sparrow-whole-section")


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


def _pixel_to_global_um(geometry, micron_to_pixel: np.ndarray):
    pixel_to_micron = np.linalg.inv(micron_to_pixel)
    return affine_transform(
        geometry,
        [
            pixel_to_micron[0, 0],
            pixel_to_micron[0, 1],
            pixel_to_micron[1, 0],
            pixel_to_micron[1, 1],
            pixel_to_micron[0, 2],
            pixel_to_micron[1, 2],
        ],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--section-id", required=True)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--reference-commit", required=True)
    parser.add_argument("--environment-receipt", required=True, type=Path)
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


def main() -> None:
    args = _parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    section = _select_section(config, args.section_id)
    parameters = config["parameters"]
    method = config["method"]
    model = config["model"]

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
    work_dir = (Path(scratch_root) / f"sparrow-{args.section_id}").resolve()
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
        raise RuntimeError("the reviewed whole-section integration requires a CUDA GPU")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name, path in (("dapi", dapi_path), ("polyt", polyt_path), ("affine", affine_path)):
        LOGGER.info("validating %s input", name)
        _validate_input(path, section[f"{name}_size"], section[f"{name}_sha256"])

    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True)
    output_temporary = output.with_suffix(output.suffix + ".partial")
    receipt_temporary = receipt_path.with_suffix(receipt_path.suffix + ".partial")
    output_temporary.unlink(missing_ok=True)
    receipt_temporary.unlink(missing_ok=True)
    published_output = False
    try:
        model_path = work_dir / model["file_name"]
        LOGGER.info("staging pinned Cellpose model")
        _download_verified(model["url"], model_path, model["size_bytes"], model["sha256"])

        dapi = tifffile.memmap(dapi_path)
        polyt = tifffile.memmap(polyt_path)
        expected_shape = tuple(section["image_shape"])
        if dapi.shape != expected_shape or polyt.shape != expected_shape:
            raise ValueError("image shape does not match the declared section")
        if dapi.dtype != polyt.dtype or dapi.ndim != 2 or dapi.dtype != np.dtype("uint16"):
            raise TypeError(
                "reviewed SPArrOW inputs must be matching two-dimensional uint16 images"
            )

        chunk_pixels = parameters["chunk_pixels"]
        chunks = (chunk_pixels, chunk_pixels)
        image = da.stack(
            [
                da.from_array(polyt, chunks=chunks, asarray=False),
                da.from_array(dapi, chunks=chunks, asarray=False),
            ],
            axis=0,
        )
        raw = Image2DModel.parse(
            image,
            dims=("c", "y", "x"),
            c_coords=["PolyT", "DAPI"],
            chunks=(1, *chunks),
            scale_factors=None,
        )
        store = work_dir / "sparrow.zarr"
        LOGGER.info("materializing z4 image")
        SpatialData(images={"z4": raw}).write(store)
        sdata = sd.read_zarr(store)

        LOGGER.info("running channel-specific SPArrOW preprocessing")
        sdata = sp.im.min_max_filtering(
            sdata,
            img_layer="z4",
            output_layer="min_max_z4",
            size_min_max_filter=[
                parameters["polyt_background_filter"],
                parameters["dapi_background_filter"],
            ],
            overwrite=True,
        )
        sdata = sp.im.enhance_contrast(
            sdata,
            img_layer="min_max_z4",
            output_layer="clahe_z4",
            contrast_clip=[
                parameters["polyt_clahe_clip"],
                parameters["dapi_clahe_clip"],
            ],
            overwrite=True,
        )
        LOGGER.info("running Cellpose segmentation on one serialized GPU worker")
        torch.cuda.set_device(0)
        with dask.config.set(scheduler="synchronous"):
            sdata = sp.im.segment(
                sdata,
                img_layer="clahe_z4",
                model=sp.im.cellpose_callable,
                depth=parameters["overlap_pixels"],
                chunks=chunk_pixels,
                pretrained_model=str(model_path),
                device="cuda:0",
                min_size=parameters["min_size"],
                diameter=parameters["diameter"],
                flow_threshold=parameters["flow_threshold"],
                cellprob_threshold=parameters["cellprob_threshold"],
                channels=[1, 2],
                output_labels_layer="labels_z4_d100",
                output_shapes_layer="boundaries_z4_d100",
                overwrite=True,
            )

        micron_to_pixel = np.loadtxt(affine_path)
        if micron_to_pixel.shape != (3, 3) or not np.allclose(micron_to_pixel[2], [0, 0, 1]):
            raise ValueError("affine must be a homogeneous 3x3 micron-to-pixel transform")
        shapes = sdata.shapes["boundaries_z4_d100"].copy().sort_index(kind="stable")
        geometries = shapes.geometry.map(
            lambda geometry: _pixel_to_global_um(geometry, micron_to_pixel)
        )
        cells = gpd.GeoDataFrame(
            {
                "cell_id": [
                    f"{args.section_id}-sparrow-{index}" for index in shapes.index.astype(str)
                ]
            },
            geometry=list(geometries),
        )
        cells.to_parquet(output_temporary, index=False)
        output_sha256 = _sha256(output_temporary)
        receipt = {
            "schema_version": 1,
            "reference_id": "sparrow",
            "reference_commit": args.reference_commit,
            "upstream_repository": method["upstream_repository"],
            "upstream_commit": method["upstream_commit"],
            "segmentation_id": method["segmentation_id"],
            "section_id": args.section_id,
            "scope": "whole_section",
            "coordinate_system": "global_um",
            "parameters": parameters,
            "inputs": {
                "affine_sha256": section["affine_sha256"],
                "dapi_sha256": section["dapi_sha256"],
                "polyt_sha256": section["polyt_sha256"],
            },
            "cellpose_model_sha256": model["sha256"],
            "bootstrap_environment_receipt_sha256": _sha256(args.environment_receipt),
            "realized_environment_receipt_sha256": _sha256(realized_receipt_path),
            "entrypoint_sha256": _sha256(Path(__file__)),
            "config_sha256": _sha256(args.config),
            "software": {
                "cellpose": cellpose.version,
                "sparrow": sp.__version__,
                "torch": torch.__version__,
            },
            "resources": config["resources"],
            "n_cells": len(cells),
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
