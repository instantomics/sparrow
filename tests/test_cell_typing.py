from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

# Load the reference-owned numerical helper without importing the task entrypoint.
spec = importlib.util.spec_from_file_location(
    "sparrow_cell_typing", Path(__file__).parents[1] / "candidate/src/mymodel/cell_typing.py"
)
cell_typing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cell_typing)


def infer(counts, reference, labels):
    return cell_typing._type_probabilities(
        sparse.csr_matrix(counts),
        sparse.csr_matrix(reference),
        labels,
        concentration=20.0,
        smoothing=0.05,
    )


def test_sparse_evidence_updates_prior_without_hard_labels():
    empty, one, more, mixed = infer(
        [[0, 0], [1, 0], [5, 0], [100_000, 100_000]], [[9, 1], [1, 9]], ("A", "B")
    )
    assert empty == {"A": 0.5, "B": 0.5}
    assert one["A"] == pytest.approx(0.88)
    assert one["A"] < more["A"] < 1
    assert mixed["A"] == pytest.approx(0.5)
    for distribution in (empty, one, more, mixed):
        assert all(np.isfinite(value) and 0 <= value <= 1 for value in distribution.values())
        assert sum(distribution.values()) == pytest.approx(1, abs=1e-8)


def test_reference_depth_and_sampling_frequency_do_not_set_type_prior():
    expected = infer([[0, 0], [2, 1]], [[9, 1], [1, 9]], ("A", "B"))
    actual = infer([[0, 0], [2, 1]], [[1, 9], [90, 10], [9, 1]], ("B", "A", "A"))
    for left, right in zip(expected, actual, strict=True):
        assert left == pytest.approx(right)


@pytest.mark.parametrize(
    "counts,reference",
    [
        (np.empty((2, 0)), np.empty((2, 0))),
        ([[0], [500]], [[10], [1]]),
        ([[0, 0], [10, 0]], [[0, 0], [0, 0]]),
        ([[0, 3], [5, 1]], [[3, 1], [30, 10]]),
    ],
)
def test_uninformative_panels_retain_label_uncertainty_not_unknown(counts, reference):
    assert infer(counts, reference, ("A", "B")) == [{"A": 0.5, "B": 0.5}] * 2


def test_type_without_panel_rna_is_not_excluded():
    empty, observed = infer([[0, 0], [1, 0]], [[9, 1], [1, 9], [0, 0]], ("A", "B", "C"))
    assert empty == pytest.approx({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
    assert observed["A"] > observed["C"] > observed["B"] > 0


def test_batched_sparse_cells_preserve_order_and_empty_rows():
    counts = np.zeros((520, 2), dtype=int)
    counts[255] = [5, 0]
    counts[256] = [0, 5]
    result = infer(counts, [[9, 1], [1, 9]], ("A", "B"))
    assert len(result) == 520
    assert result[255]["A"] > 0.99
    assert result[256]["B"] > 0.99
    assert result[254] == result[257] == result[-1] == {"A": 0.5, "B": 0.5}
