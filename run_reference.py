from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

_MODEL_URL = "https://www.cellpose.org/models/cytotorch_0"
_MODEL_SHA256 = "6a852487b98a3ad91e4e86c969cac520aaac13c609288aad8bd01d4cf76370c6"
_MODEL_SIZE = 26_563_614
_TERMINAL_STATUSES = {"succeeded", "failed", "timed_out", "cancelled", "orphaned"}
_SPARROW_BASE = (
    "https://raw.githubusercontent.com/saeyslab/napari-sparrow/"
    "fcdb27cabe51cbb8b3f0f6d2b2a58530e9fbc43f/"
)
_ANNOTATION_FILES = (
    (
        "src/sparrow/table/_annotation.py",
        "annotation.txt",
        34576,
        "c18693179f974ba57f32d3e86f2da1d1f4c78f0afb1c25a8803f0390c8e5533c",
    ),
    (
        "src/sparrow/utils/_keys.py",
        "keys.txt",
        483,
        "b00be413204d7756434c22dc444dd5bc0dd4aae77e7570ee8a249e13e6dae0b0",
    ),
    (
        "LICENSE",
        "LICENSE.txt",
        8403,
        "fe175634aa738f5c9c33be398d8c489f79eec6e68a33852e32b2761a5049ef24",
    ),
)


def _stage_annotation(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source, name, size, digest in _ANNOTATION_FILES:
        request = urllib.request.Request(
            _SPARROW_BASE + source, headers={"User-Agent": "iomix-sparrow-reference"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(size + 1)
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"SPArrOW source does not match its declared identity: {source}")
        (destination / name).write_bytes(data)


def _download_model(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    digest = hashlib.sha256()
    size = 0
    request = urllib.request.Request(_MODEL_URL, headers={"User-Agent": "iomix-sparrow-reference"})
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as stream,
        ):
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > _MODEL_SIZE:
                    raise ValueError("Cellpose model download exceeds its declared size")
                digest.update(chunk)
                stream.write(chunk)
        if size != _MODEL_SIZE or digest.hexdigest() != _MODEL_SHA256:
            raise ValueError("Cellpose model download does not match its declared identity")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run(tools, context):
    model_path = Path(context.candidate_path) / "main/src/mymodel/cytotorch_0"
    _download_model(model_path)
    _stage_annotation(model_path.parent / "sparrow_native")

    candidate_label = context.reference_id
    tools.call("freeze_candidate", {"candidate_label": candidate_label})

    validation = tools.call("validate_model", {"candidate_label": candidate_label})
    if validation.get("valid") is not True:
        raise RuntimeError(f"candidate validation failed: {validation!r}")

    evaluation = tools.call(
        "start_evaluation",
        {"candidate_label": candidate_label, "profile_id": "validation"},
    )
    job_id = evaluation["job_id"]
    for _ in range(5):
        current = tools.call("wait_job", {"job_id": job_id, "seconds": 300})
        if current["status"] in _TERMINAL_STATUSES:
            break

    completed = tools.call("inspect_job", {"job_id": job_id, "view": "summary", "detail": "full"})
    if completed["status"] != "succeeded":
        raise RuntimeError(f"reference evaluation did not succeed: {completed!r}")
    return completed
