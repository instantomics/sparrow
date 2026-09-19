import importlib.metadata
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_reference


class _Tools:
    def __init__(self):
        self.calls = []

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "validate_model":
            return {"valid": True}
        if name == "start_evaluation":
            return {"job_id": "evaluation"}
        if name in {"wait_job", "inspect_job"}:
            return {"status": "succeeded"}
        return {}


@pytest.mark.parametrize("tamper", [False, True])
def test_native_source_identity_is_checked_before_freezing(monkeypatch, tmp_path: Path, tamper):
    distribution = importlib.metadata.distribution("sparrow")
    sources = run_reference._ANNOTATION_FILES[:2]
    monkeypatch.setattr(run_reference, "_ANNOTATION_FILES", sources)
    data = {
        run_reference._SPARROW_BASE + source: distribution.locate_file(
            source.removeprefix("src/")
        ).read_bytes()
        for source, _, _, _ in sources
    }
    if tamper:
        url = run_reference._SPARROW_BASE + sources[0][0]
        original = data[url]
        data[url] = bytes([original[0] ^ 1]) + original[1:]
    monkeypatch.setattr(
        run_reference.urllib.request,
        "urlopen",
        lambda request, timeout: BytesIO(data[request.full_url]),
    )

    def stage_model(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test-model")

    monkeypatch.setattr(run_reference, "_download_model", stage_model)
    candidate = tmp_path / "main/src/mymodel"

    class Tools(_Tools):
        def call(self, name, arguments):
            if name == "freeze_candidate":
                for source, destination, _, _ in sources:
                    assert (candidate / "sparrow_native" / destination).read_bytes() == data[
                        run_reference._SPARROW_BASE + source
                    ]
            return super().call(name, arguments)

    tools = Tools()
    context = SimpleNamespace(candidate_path=str(tmp_path), reference_id="sparrow")
    if tamper:
        with pytest.raises(ValueError, match="declared identity"):
            run_reference.run(tools, context)
        assert not tools.calls
    else:
        run_reference.run(tools, context)
        assert tools.calls[0][0] == "freeze_candidate"
