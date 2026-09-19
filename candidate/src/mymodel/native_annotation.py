from __future__ import annotations

import ast
import logging
import re
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
from anndata import AnnData

NATIVE_ROOT = Path(__file__).with_name("sparrow_native")


@lru_cache(maxsize=1)
def _annotation_functions():
    """Load the unchanged, non-plotting core of upstream score_genes_iter.

    The exact upstream files and license are admitted before candidate freezing.
    Selecting their function ASTs avoids importing the incompatible SpatialData
    orchestration stack; no annotation function body is rewritten.
    """
    namespace = {
        "np": np,
        "pd": pd,
        "Path": Path,
        "re": re,
        "MappingProxyType": MappingProxyType,
        "log": logging.getLogger("sparrow.table._annotation"),
    }
    keys = ast.parse((NATIVE_ROOT / "keys.txt").read_text())
    annotation = ast.parse((NATIVE_ROOT / "annotation.txt").read_text())
    constants = {"_ANNOTATION_KEY", "_CLEANLINESS_KEY", "_UNKNOWN_CELLTYPE_KEY"}
    functions = {"_annotate_celltype_iter", "_annotate_celltype_weighted"}
    nodes = [
        node
        for node in keys.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets)
    ]
    nodes += [
        node
        for node in annotation.body
        if isinstance(node, ast.FunctionDef) and node.name in functions
    ]
    future = ast.parse("from __future__ import annotations").body
    module = ast.Module(body=future + nodes, type_ignores=[])
    exec(compile(module, str(NATIVE_ROOT / "annotation.txt"), "exec"), namespace)
    return namespace


def annotation_scores(scaled, genes, markers, *, iterations):
    """Run native iterative scoring with continuous scores and no plotting."""
    namespace = _annotation_functions()
    # Upstream sanitizes label names. Opaque internal IDs avoid collisions and
    # preserve punctuation/whitespace in the candidate-visible vocabulary.
    type_ids = [f"type_{index}" for index in range(markers.shape[1])]
    adata = AnnData(
        np.asarray(scaled, dtype=np.float64),
        var=pd.DataFrame(index=list(genes)),
    )
    marker_table = pd.DataFrame(markers.astype(int), index=list(genes), columns=type_ids)
    annotated, _, _ = namespace["_annotate_celltype_iter"](
        adata,
        marker_table,
        n_iter=iterations,
        min_score=None,
        scaling="Nmarkers",
        calculate_umap=False,
        calculate_neighbors=False,
    )
    scores = annotated.obs[type_ids].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("SPArrOW annotation returned non-finite marker scores")
    return scores
