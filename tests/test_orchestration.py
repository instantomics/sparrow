import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1]))

import run_reference  # noqa: E402


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


def test_model_is_staged_in_selected_candidate(monkeypatch, tmp_path: Path) -> None:
    staged = []
    monkeypatch.setattr(run_reference, "_download_model", staged.append)

    run_reference.run(_Tools(), SimpleNamespace(candidate_path=str(tmp_path), reference_id="sparrow"))

    assert staged == [tmp_path / "main/src/mymodel/cytotorch_0"]
