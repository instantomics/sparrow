from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from shapely import STRtree, points
from shapely.geometry import Polygon

from .native_annotation import annotation_scores


@dataclass(frozen=True)
class MarkerTypingConfig:
    typing_marker_count: int = 20
    typing_min_log2_fold_change: float = 1.0
    typing_min_detection_fraction: float = 0.25
    typing_iterations: int = 5
    typing_prior_count: float = 8.0
    typing_seed: int = 0

    def __post_init__(self):
        for name in ("typing_marker_count", "typing_iterations"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.typing_seed) is not int or self.typing_seed < 0:
            raise ValueError("typing_seed must be a nonnegative integer")
        for name in (
            "typing_min_log2_fold_change",
            "typing_min_detection_fraction",
            "typing_prior_count",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if self.typing_min_detection_fraction > 1:
            raise ValueError("typing_min_detection_fraction must not exceed one")


def type_cells(cells, transcripts, reference, *, config: MarkerTypingConfig):
    """Type the final SPArrOW polygons using only the field's visible reference."""
    if not reference.cell_type_labels:
        raise ValueError("SPArrOW typing requires candidate-visible reference labels")
    shared = reference.matched_expression(transcripts.gene_ids)
    counts = _assigned_counts(cells, transcripts, shared.gene_ids)
    probabilities = _type_probabilities(
        counts,
        shared.counts,
        reference.cell_type_labels,
        reference.cell_ids,
        shared.gene_ids,
        config=config,
    )
    return tuple(
        replace(cell, type_probabilities=values)
        for cell, values in zip(cells, probabilities, strict=True)
    )


def _assigned_counts(cells, transcripts, shared_genes) -> sparse.csr_matrix:
    """Count strict polygon-interior transcripts, bounded in spatial-query memory."""
    shape = (len(cells), len(shared_genes))
    counts = sparse.csr_matrix(shape, dtype=np.int64)
    if not cells or not shared_genes:
        return counts
    shared_index = {gene: index for index, gene in enumerate(shared_genes)}
    gene_projection = np.asarray(
        [shared_index.get(gene, -1) for gene in transcripts.gene_ids], dtype=np.int64
    )
    tree = STRtree([Polygon(cell.vertices, cell.interior_rings) for cell in cells])
    for start in range(0, len(transcripts.coordinates), 65_536):
        stop = start + 65_536
        pairs = tree.query(points(transcripts.coordinates[start:stop]), predicate="within")
        genes = gene_projection[transcripts.gene_index[start:stop][pairs[0]]]
        keep = genes >= 0
        counts += sparse.csr_matrix(
            (np.ones(keep.sum(), dtype=np.int64), (pairs[1, keep], genes[keep])), shape=shape
        )
    return counts


def _type_probabilities(counts, reference_counts, labels, cell_ids, genes, *, config):
    """Reference-derived markers, native SPArrOW scoring, held-out calibration."""
    names, label_index = np.unique(labels, return_inverse=True)
    n_types = len(names)
    prior = np.full((counts.shape[0], n_types), 1.0 / n_types)
    if counts.shape[0] == 0 or len(genes) < 2 or n_types < 2 or reference_counts.sum() == 0:
        return _distributions(names, prior)

    train, heldout = _split_reference(reference_counts, label_index, cell_ids, config.typing_seed)
    train_counts = reference_counts[train]
    markers = _derive_markers(train_counts, label_index[train], n_types, genes, config)
    # Types without distinguishing markers or an independent calibration cell
    # retain their prior mass. No data deficit establishes an unknown cell type.
    supported = np.flatnonzero(
        markers.any(axis=0) & (np.bincount(label_index[heldout], minlength=n_types) > 0)
    )
    if len(supported) < 2:
        return _distributions(names, prior)
    selected = markers[:, supported].any(axis=1)
    markers = markers[selected][:, supported]
    selected_genes = np.asarray(genes)[selected]
    heldout = heldout[np.isin(label_index[heldout], supported)]
    calibration, calibration_labels = _calibration_counts(
        reference_counts[heldout],
        np.searchsorted(supported, label_index[heldout]),
        config.typing_seed,
    )

    # Fit preprocessing on the marker-discovery cells, never the held-out labels.
    training = _log_normalize(train_counts)[:, selected]
    mean = np.asarray(training.mean(axis=0)).ravel()
    variance = np.asarray(training.power(2).mean(axis=0)).ravel() - mean**2
    scale = np.sqrt(np.maximum(variance, 1e-6))
    combined = sparse.vstack((calibration, counts), format="csr")
    depth = np.asarray(combined[:, selected].sum(axis=1)).ravel()
    informative = depth > 0
    scores = np.zeros((combined.shape[0], len(supported)))
    if informative.any():
        scaled = (_log_normalize(combined[informative])[:, selected].toarray() - mean) / scale
        # Calibration examples and target cells are scored together, with no
        # labels passed to the native iterative annotation routine.
        scores[informative] = annotation_scores(
            scaled, selected_genes, markers, iterations=config.typing_iterations
        )
    n_calibration = calibration.shape[0]
    temperature = _calibrate_temperature(
        scores[:n_calibration], calibration_labels, depth[:n_calibration], config.typing_prior_count
    )
    conditional = _score_probabilities(
        scores[n_calibration:], depth[n_calibration:], temperature, config.typing_prior_count
    )
    prior[:, supported] = conditional * (len(supported) / n_types)
    return _distributions(names, prior)


def _distributions(names, probabilities):
    return [dict(zip(names.tolist(), row.tolist(), strict=True)) for row in probabilities]


def _split_reference(counts, labels, cell_ids, seed):
    """Stable disjoint discovery/calibration split, independent of row order."""
    totals = np.asarray(counts.sum(axis=1)).ravel()
    train, heldout = [], []
    for label in np.unique(labels):
        rows = np.flatnonzero((labels == label) & (totals > 0)).tolist()
        rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{cell_ids[row]}".encode()).digest())
        n_calibration = min(32, len(rows) // 3)
        heldout.extend(rows[:n_calibration])
        train.extend(rows[n_calibration:])
    return np.asarray(train, dtype=int), np.asarray(heldout, dtype=int)


def _derive_markers(counts, labels, n_types, genes, config):
    """Positive one-vs-rest enrichment, with a balanced rest and detection gate."""
    sizes = np.bincount(labels, minlength=n_types)
    inverse_sizes = np.divide(1.0, sizes, out=np.zeros(n_types), where=sizes > 0)
    membership = sparse.csr_matrix(
        (inverse_sizes[labels], (labels, np.arange(len(labels)))), shape=(n_types, len(labels))
    )
    totals = np.asarray(counts.sum(axis=1)).ravel()
    fractions = sparse.diags(1.0 / totals) @ counts
    profiles = (membership @ fractions).toarray()
    detection = (membership @ (counts > 0)).toarray()
    rest = (profiles.sum(axis=0) - profiles) / max(1, np.count_nonzero(sizes) - 1)
    enrichment = np.log2((profiles + 1e-6) / (rest + 1e-6))
    markers = np.zeros((len(genes), n_types), dtype=bool)
    for label in range(n_types):
        eligible = np.flatnonzero(
            (enrichment[label] >= config.typing_min_log2_fold_change)
            & (detection[label] >= config.typing_min_detection_fraction)
            & (sizes[label] >= 2)
        )
        order = sorted(
            eligible,
            key=lambda gene: (-enrichment[label, gene] * detection[label, gene], genes[gene]),
        )
        markers[order[: config.typing_marker_count], label] = True
    return markers


def _log_normalize(counts):
    totals = np.asarray(counts.sum(axis=1)).ravel()
    factors = np.divide(10_000.0, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    values = (sparse.diags(factors) @ counts).tocsr()
    np.log1p(values.data, out=values.data)
    return values


def _calibration_counts(counts, labels, seed):
    """Thin held-out cells without replacement to cover sparse RNA depths."""
    rng = np.random.default_rng(seed)
    rows, targets = [], []
    for index, label in enumerate(labels):
        original = counts[index].toarray().ravel().astype(np.int64)
        total = int(original.sum())
        for depth in sorted({min(total, depth) for depth in (4, 16, 64, 256)}):
            sampled = rng.multivariate_hypergeometric(original, depth)
            rows.append(sparse.csr_matrix(sampled[None, :]))
            targets.append(label)
    return sparse.vstack(rows, format="csr"), np.asarray(targets, dtype=int)


def _score_probabilities(scores, depth, temperature, prior_count):
    weight = np.asarray(depth, dtype=float)[:, None] / (np.asarray(depth)[:, None] + prior_count)
    return weight * softmax(scores / temperature, axis=1) + (1 - weight) / scores.shape[1]


def _calibrate_temperature(scores, labels, depth, prior_count):
    """Fit a single temperature by type-balanced held-out log loss."""
    sizes = np.bincount(labels, minlength=scores.shape[1])
    weights = 1.0 / sizes[labels]

    def loss(log_temperature):
        probabilities = _score_probabilities(scores, depth, np.exp(log_temperature), prior_count)
        correct = probabilities[np.arange(len(labels)), labels]
        return float(np.average(-np.log(np.maximum(correct, 1e-15)), weights=weights))

    fit = minimize_scalar(loss, bounds=(math.log(0.05), math.log(1000)), method="bounded")
    if not fit.success:
        raise RuntimeError("SPArrOW score temperature calibration failed")
    # Do not force a confident mapping when the held-out evidence fails even
    # to improve on the equal-type prior.
    return float(np.exp(fit.x)) if fit.fun < math.log(scores.shape[1]) - 1e-8 else math.inf
