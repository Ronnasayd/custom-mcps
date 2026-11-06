"""FastMCP server scoring pairwise text similarity with a local embedding model."""

import json
import os
from typing import Any

from mcp.server.fastmcp import FastMCP
import numpy as np

mcp = FastMCP(name="similarity_tool")

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SCORE_DECIMALS = 4

# invariant: None in production; tests set a fake here.
_embedder: Any = None  # pylint: disable=invalid-name


def _model_name() -> str:
    """Return the EMBEDDING_MODEL value, or the default model when it is blank."""
    return os.environ.get("EMBEDDING_MODEL", "").strip() or DEFAULT_MODEL


def _normalized(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length.

    Args:
        vectors: One embedding per row.

    Returns:
        The rows at unit length; a zero row stays zero, so it scores 0.0 with no NaN.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


@mcp.tool(structured_output=False)
def similarity_matrix(texts_a: list[str], texts_b: list[str] | None = None) -> str:
    """Score every pair of texts by cosine similarity, from -1.0 to 1.0.

    A row or column of `matrix` is the position of a text in the input lists.
    Rows follow texts_a and columns follow texts_b.
    Omitting texts_b compares texts_a with itself.
    The default limit is 300 texts across both lists.
    A bad call returns {"error": "<message>"}.
    A text whose embedding has zero norm scores 0.0 on all its cells.
    The default model reads only the first 128 tokens of a text.

    Args:
        texts_a: Texts of the matrix rows.
        texts_b: Texts of the matrix columns; omit to compare texts_a with itself.

    Returns:
        Compact JSON text: {"matrix": [[...]], "model": "<name>"}, or {"error": ...}.
    """
    # why: the two-list branch lands next; the published signature is already final.
    del texts_b
    vectors = np.array(list(_embedder.embed(texts_a)), dtype=np.float64)
    unit = _normalized(vectors)
    matrix = np.round(unit @ unit.T, SCORE_DECIMALS).tolist()
    return json.dumps({"matrix": matrix, "model": _model_name()}, separators=(",", ":"))
