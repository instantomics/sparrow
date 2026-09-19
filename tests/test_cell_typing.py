from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from mymodel import cell_typing
from mymodel.native_annotation import annotation_scores

CONFIG = cell_typing.MarkerTypingConfig()
GENES = ("A-marker", "B-marker", "housekeeping")
REFERENCE = np.array([[90, 0, 10]] * 6 + [[0, 90, 10]] * 6)
LABELS = ("T cell",) * 6 + ("T/cell",) * 6
CELL_IDS = tuple(f"ref-{index}" for index in range(len(LABELS)))


def infer(counts, reference=REFERENCE, labels=LABELS, genes=GENES, cell_ids=None):
    if cell_ids is None:
        cell_ids = tuple(f"ref-{index}" for index in range(len(labels)))
    return cell_typing._type_probabilities(
        sparse.csr_matrix(counts),
        sparse.csr_matrix(reference),
        labels,
        cell_ids,
        genes,
        config=CONFIG,
    )


def test_markers_select_enrichment_and_reject_housekeeping_and_rare_reads():
    counts = sparse.csr_matrix([[90, 0, 10, 1]] + [[90, 0, 10, 0]] * 5 + [[0, 90, 10, 0]] * 6)
    labels = np.array([0] * 6 + [1] * 6)
    markers = cell_typing._derive_markers(counts, labels, 2, (*GENES, "rare"), CONFIG)
    np.testing.assert_array_equal(markers, [[1, 0], [0, 1], [0, 0], [0, 0]])
    stricter = replace(CONFIG, typing_min_log2_fold_change=30)
    assert not cell_typing._derive_markers(counts, labels, 2, (*GENES, "rare"), stricter).any()


def test_native_iterative_scores_use_supplied_marker_table(native_annotation_source):
    expression = np.array([[2, -2], [-2, 2], [1, -1], [-1, 1]], dtype=float)
    markers = np.eye(2, dtype=bool)
    scores = annotation_scores(expression, ("a", "b"), markers, iterations=5)
    swapped = annotation_scores(expression, ("a", "b"), markers[:, ::-1], iterations=5)
    np.testing.assert_allclose(scores[:, 0] - scores[:, 1], [4, -4, 2, -2])
    np.testing.assert_allclose(swapped, scores[:, ::-1])


def test_sparse_cells_and_literal_labels_through_native_annotation(native_annotation_source):
    empty, one, dense, other, no_marker = infer(
        [[0, 0, 0], [1, 0, 0], [90, 0, 10], [0, 90, 10], [0, 0, 100]]
    )
    assert empty == no_marker == {"T cell": 0.5, "T/cell": 0.5}
    assert 0.5 < one["T cell"] < 0.6
    assert dense["T cell"] > 0.9
    assert other["T/cell"] > 0.9
    for distribution in (empty, one, dense, other, no_marker):
        assert set(distribution) == {"T cell", "T/cell"}
        assert all(np.isfinite(value) and 0 <= value <= 1 for value in distribution.values())
        assert sum(distribution.values()) == pytest.approx(1, abs=1e-8)


def test_holdout_labels_calibrate_but_cannot_choose_markers(native_annotation_source, monkeypatch):
    _, encoded = np.unique(LABELS, return_inverse=True)
    train, heldout = cell_typing._split_reference(
        sparse.csr_matrix(REFERENCE), encoded, CELL_IDS, 0
    )
    assert not set(train) & set(heldout)
    assert set(train) | set(heldout) == set(range(len(LABELS)))
    tables = []
    native = cell_typing.annotation_scores

    def record(*args, **kwargs):
        tables.append(args[2].copy())
        return native(*args, **kwargs)

    monkeypatch.setattr(cell_typing, "annotation_scores", record)
    good = infer([[90, 0, 10]])[0]
    corrupted = REFERENCE.copy()
    corrupted[heldout] = corrupted[heldout][:, [1, 0, 2]]
    uncertain = infer([[90, 0, 10]], corrupted)[0]
    np.testing.assert_array_equal(tables[0], tables[1])
    assert good["T cell"] > 0.9
    assert uncertain == {"T cell": 0.5, "T/cell": 0.5}


def test_reference_row_order_does_not_change_split_or_probabilities(native_annotation_source):
    order = np.arange(len(LABELS))[::-1]
    counts = [[30, 0, 5], [0, 30, 5]]
    expected = infer(counts)
    actual = infer(
        counts, REFERENCE[order], np.asarray(LABELS)[order], cell_ids=np.asarray(CELL_IDS)[order]
    )
    for left, right in zip(expected, actual, strict=True):
        assert left == pytest.approx(right)


@pytest.mark.parametrize(
    "counts,reference,labels,genes",
    [
        (np.empty((2, 0)), np.empty((6, 0)), ("A", "B") * 3, ()),
        ([[0], [500]], [[10], [1]] * 3, ("A", "B") * 3, ("a",)),
        ([[0, 0], [10, 0]], [[0, 0]] * 6, ("A", "B") * 3, ("a", "b")),
        ([[1, 0], [0, 1]], [[90, 1], [1, 90]] * 2, ("A", "B") * 2, ("a", "b")),
        ([[1, 0], [0, 1]], [[3, 1], [30, 10]] * 3, ("A", "B") * 3, ("a", "b")),
    ],
)
def test_unsupported_panels_and_tiny_references_retain_prior(counts, reference, labels, genes):
    assert infer(counts, reference, labels, genes) == [{"A": 0.5, "B": 0.5}] * 2


def test_type_without_distinguishing_markers_keeps_prior_mass(native_annotation_source):
    reference = np.vstack((REFERENCE, [[45, 45, 10]] * 6))
    result = infer([[90, 0, 10], [0, 90, 10]], reference, LABELS + ("ambiguous",) * 6)
    for probabilities in result:
        assert probabilities["ambiguous"] == pytest.approx(1 / 3)
        assert sum(probabilities.values()) == pytest.approx(1)
    assert result[0]["T cell"] > result[0]["T/cell"]
    assert result[1]["T/cell"] > result[1]["T cell"]


def test_calibration_thinning_preserves_depth_and_never_invents_transcripts():
    original = sparse.csr_matrix([[100, 0, 100], [0, 0, 5]])
    thinned, labels = cell_typing._calibration_counts(original, np.array([0, 1]), 0)
    assert set(np.asarray(thinned[labels == 0].sum(axis=1)).ravel()) == {4, 16, 64, 200}
    assert set(np.asarray(thinned[labels == 1].sum(axis=1)).ravel()) == {4, 5}
    for row, label in zip(thinned.toarray(), labels, strict=True):
        assert np.all(row <= original[label].toarray().ravel())
