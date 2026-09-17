from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

_MODEL_URL = "https://www.cellpose.org/models/cytotorch_0"
_MODEL_SHA256 = "6a852487b98a3ad91e4e86c969cac520aaac13c609288aad8bd01d4cf76370c6"
_MODEL_SIZE = 26_563_614
_TERMINAL_STATUSES = {"succeeded", "failed", "timed_out", "cancelled", "orphaned"}


def _download_model(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    digest = hashlib.sha256()
    size = 0
    request = urllib.request.Request(_MODEL_URL, headers={"User-Agent": "iomix-sparrow-reference"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
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
