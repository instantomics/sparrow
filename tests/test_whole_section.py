from __future__ import annotations

import copy
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

import whole_section

ROOT = Path(__file__).parents[1]


def _receipt() -> dict:
    return {
        "target": {"kind": "reference", "id": "sparrow"},
        "profile": "authoring",
        "python": {
            "abi": "cpython-311",
            "version": "3.11.5",
            "executable": "/modules/python3.11",
        },
        "packages": [
            {
                "project_id": "reference:sparrow",
                "commit": "abc123",
                "dirty": False,
                "lock_digest": {"algorithm": "sha256", "value": "lock"},
                "pyproject_digest": {"algorithm": "sha256", "value": "project"},
            }
        ],
    }


def test_realized_environment_must_match_staged_reference_and_python() -> None:
    staged = _receipt()
    realized = copy.deepcopy(staged)
    assert whole_section._validate_realized_environment(staged, realized)["commit"] == "abc123"

    realized["python"]["abi"] = "cpython-313"
    with pytest.raises(ValueError, match="Python abi"):
        whole_section._validate_realized_environment(staged, realized)


def test_realized_environment_rejects_dirty_reference() -> None:
    staged = _receipt()
    realized = copy.deepcopy(staged)
    realized["packages"][0]["dirty"] = True

    with pytest.raises(ValueError, match="clean committed reference"):
        whole_section._validate_realized_environment(staged, realized)


def test_candidate_source_binding_rejects_drift(tmp_path: Path) -> None:
    config = tomllib.loads((ROOT / "whole_section.toml").read_text())
    source = ROOT / "candidate/src"
    staged = tmp_path / "candidate_source"
    shutil.copytree(source / "mymodel", staged / "mymodel")
    shutil.copyfile(ROOT / "candidate/iomix_candidate.json", staged / "iomix_candidate.json")

    observed = whole_section._validate_candidate_source(staged, config["candidate_source"])
    assert set(observed) == {
        "iomix_candidate.json",
        "mymodel/__init__.py",
        "mymodel/method.py",
        "mymodel/cell_typing.py",
        "mymodel/native_annotation.py",
    }
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys,tomllib; from pathlib import Path; "
                "sys.path.insert(0, sys.argv[1]); import whole_section as w; "
                "config=tomllib.load(open(Path(sys.argv[1])/'whole_section.toml','rb')); "
                "method=w._load_candidate_method(Path(sys.argv[2]),config['candidate_source']); "
                "assert Path(method.__file__).resolve() == "
                "(Path(sys.argv[2])/'mymodel/method.py').resolve()"
            ),
            str(ROOT),
            str(staged),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    with (staged / "mymodel/method.py").open("ab") as handle:
        handle.write(b"\n# drift\n")
    with pytest.raises(ValueError, match="staged candidate source changed: mymodel/method.py"):
        whole_section._validate_candidate_source(staged, config["candidate_source"])


def test_halo_tiles_preserve_windows_and_clip_to_disjoint_cores() -> None:
    tiles, pixelization = whole_section._tile_plan(
        (300, 300), scale_x=1.0, scale_y=1.0, core_um=100.0, halo_um=25.0
    )
    assert len(tiles) == 9
    left = next(tile for tile in tiles if (tile.row, tile.column) == (0, 0))
    right = next(tile for tile in tiles if (tile.row, tile.column) == (0, 1))
    last = next(tile for tile in tiles if (tile.row, tile.column) == (0, 2))
    assert (left.window_x0, left.window_x1) == (0, 150)
    assert (right.window_x0, right.window_x1) == (75, 225)
    assert (last.window_x0, last.window_x1) == (150, 300)

    affine = np.eye(3)
    seam_cell = Polygon([(99.5, 40), (100.5, 40), (100.5, 50), (99.5, 50)])
    left_piece = seam_cell.intersection(whole_section._core_geometry(left, affine))
    right_piece = seam_cell.intersection(whole_section._core_geometry(right, affine))
    assert left_piece.area == pytest.approx(5.0)
    assert right_piece.area == pytest.approx(5.0)
    assert left_piece.intersection(right_piece).area == 0
    assert pixelization == {
        "core_pixels_x": 100,
        "core_pixels_y": 100,
        "halo_pixels_x": 25,
        "halo_pixels_y": 25,
    }


def test_tile_coordinates_apply_declared_global_affine() -> None:
    tiles, _ = whole_section._tile_plan(
        (300, 300), scale_x=2.0, scale_y=4.0, core_um=50.0, halo_um=10.0
    )
    tile = next(tile for tile in tiles if (tile.row, tile.column) == (1, 1))
    affine = np.asarray([[2.0, 0.0, -20.0], [0.0, 4.0, -40.0], [0.0, 0.0, 1.0]])

    origin, pixel_size, bounds = whole_section._tile_coordinates(tile, affine)

    assert origin == pytest.approx((50.0, 15.0))
    assert pixel_size == pytest.approx((0.5, 0.25))
    assert np.allclose(affine @ np.asarray([origin[0], origin[1], 1]), [80, 20, 1])
    assert bounds == pytest.approx((50.0, 15.0, 120.0, 85.0))
