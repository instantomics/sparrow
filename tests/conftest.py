import hashlib
import importlib.metadata
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "candidate/src"))

from mymodel import native_annotation  # noqa: E402
from run_reference import _ANNOTATION_FILES  # noqa: E402


@pytest.fixture
def native_annotation_source(monkeypatch, tmp_path):
    """Exercise the real pinned upstream code, without downloads in tests."""
    distribution = importlib.metadata.distribution("sparrow")
    for source, destination, size, digest in _ANNOTATION_FILES[:2]:
        data = distribution.locate_file(source.removeprefix("src/")).read_bytes()
        assert len(data) == size and hashlib.sha256(data).hexdigest() == digest
        (tmp_path / destination).write_bytes(data)
    monkeypatch.setattr(native_annotation, "NATIVE_ROOT", tmp_path)
    native_annotation._annotation_functions.cache_clear()
    yield
    native_annotation._annotation_functions.cache_clear()
