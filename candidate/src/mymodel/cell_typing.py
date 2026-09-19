from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import sparse
from scipy.special import gammaln, softmax
from shapely import STRtree, points
from shapely.geometry import Polygon


def type_cells(cells, transcripts, reference, *, concentration: float, smoothing: float):
    """Type the final SPArrOW polygons using only the field's visible reference."""
    if not reference.cell_type_labels:
        raise ValueError("SPArrOW typing requires candidate-visible reference labels")
    shared = reference.matched_expression(transcripts.gene_ids)
    counts = _assigned_counts(cells, transcripts, shared.gene_ids)
    probabilities = _type_probabilities(
        counts,
        shared.counts,
        reference.cell_type_labels,
        concentration=concentration,
        smoothing=smoothing,
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


def _type_probabilities(counts, reference_counts, labels, *, concentration, smoothing):
    """Equal-type-prior Dirichlet-multinomial posterior on shared-panel counts."""
    names, label_index = np.unique(labels, return_inverse=True)
    n_types = len(names)
    n_genes = counts.shape[1]
    if n_genes == 0 or reference_counts.sum() == 0:
        return [dict.fromkeys(names.tolist(), 1.0 / n_types) for _ in range(counts.shape[0])]

    # Normalize each reference cell before averaging so library size does not
    # determine its influence, and reference sampling does not set tissue priors.
    totals = np.asarray(reference_counts.sum(axis=1)).ravel()
    nonempty = totals > 0
    inverse = np.divide(1.0, totals, out=np.zeros_like(totals, dtype=float), where=nonempty)
    membership = sparse.csr_matrix(
        (inverse, (label_index, np.arange(len(labels)))), shape=(n_types, len(labels))
    )
    profiles = (membership @ reference_counts).toarray()
    type_sizes = np.bincount(label_index[nonempty], minlength=n_types)
    supported = type_sizes > 0
    profiles[supported] /= type_sizes[supported, None]
    # A type with no shared-panel RNA remains possible with a neutral profile.
    profiles[~supported] = profiles[supported].mean(axis=0)
    profiles = (1 - smoothing) * profiles + smoothing / n_genes
    alpha = concentration * profiles / profiles.sum(axis=1, keepdims=True)

    result = []
    for start in range(0, counts.shape[0], 256):
        batch = counts[start : start + 256].tocoo()
        logits = np.zeros((batch.shape[0], n_types))
        for type_index in range(n_types):
            a = alpha[type_index, batch.col]
            # The multinomial coefficient and total-concentration terms are
            # identical across types and cancel in posterior normalization.
            logits[:, type_index] = np.bincount(
                batch.row,
                weights=gammaln(a + batch.data) - gammaln(a),
                minlength=batch.shape[0],
            )
        for probabilities in softmax(logits, axis=1):
            result.append(dict(zip(names.tolist(), probabilities.tolist(), strict=True)))
    return result
