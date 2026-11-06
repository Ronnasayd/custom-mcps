"""FastMCP server scoring pairwise text similarity with a local embedding model."""

import contextlib
import functools
import json
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP
import numpy as np

mcp = FastMCP(name="similarity_tool")

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CACHE_DIR = "~/.cache/fastembed"
SCORE_DECIMALS = 4

# invariant: None in production; tests set a fake here.
_embedder: Any = None  # pylint: disable=invalid-name


def _model_name() -> str:
    """Return the EMBEDDING_MODEL value, or the default model when it is blank."""
    return os.environ.get("EMBEDDING_MODEL", "").strip() or DEFAULT_MODEL


def _cache_dir() -> str:
    """Return the expanded FASTEMBED_CACHE_PATH, or the default when it is blank."""
    value = os.environ.get("FASTEMBED_CACHE_PATH", "").strip()
    return os.path.expanduser(value or DEFAULT_CACHE_DIR)


# why: fastembed is optional, so its embedder type cannot be imported at module level.
@functools.cache
def _load_embedder(model: str) -> Any:  # noqa: ANN401
    """Build the embedder for a model once per process.

    Args:
        model: Model name to load.

    Returns:
        The loaded ``fastembed.TextEmbedding``.
    """
    # hazard: the client reads stdout as the MCP stream, so a print here corrupts it.
    with contextlib.redirect_stdout(sys.stderr):
        # pylint: disable-next=import-outside-toplevel,import-error
        import fastembed  # type: ignore[import-not-found,unused-ignore]

        return fastembed.TextEmbedding(model_name=model, cache_dir=_cache_dir())


def _get_embedder(model: str) -> Any:  # noqa: ANN401
    """Return the injected embedder, or the cached one for a model.

    Args:
        model: Model name to load when no embedder is injected.

    Returns:
        An object with an ``embed`` method.
    """
    return _embedder if _embedder is not None else _load_embedder(model)


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
    texts = texts_a if texts_b is None else texts_a + texts_b
    model = _model_name()
    vectors = np.array(list(_get_embedder(model).embed(texts)), dtype=np.float64)
    unit = _normalized(vectors)
    columns = unit if texts_b is None else unit[len(texts_a) :]
    matrix = np.round(unit[: len(texts_a)] @ columns.T, SCORE_DECIMALS).tolist()
    return json.dumps({"matrix": matrix, "model": model}, separators=(",", ":"))


def main() -> None:
    """Load the model, then run the similarity MCP server over stdio."""
    model = _model_name()
    try:
        _load_embedder(model)
    except ImportError as exc:
        sys.exit(
            f"similarity-mcp: cannot import fastembed: {exc}. "
            "Start the server with: uv run --extra embeddings similarity-mcp"
        )
    # why: a load fails in many ways (network, disk, bad model); all end the startup.
    except Exception as exc:  # pylint: disable=broad-exception-caught
        sys.exit(f"similarity-mcp: could not load model {model}: {exc}")
    sys.stderr.write("similarity-mcp: model ready\n")
    mcp.run()


if __name__ == "__main__":
    main()
