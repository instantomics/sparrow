from __future__ import annotations

import copy

import pytest

import whole_section


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
