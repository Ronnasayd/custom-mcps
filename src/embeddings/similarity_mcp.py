"""FastMCP server scoring pairwise text similarity with a local embedding model."""

import contextlib
import functools
import json
import os
import sys
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
import numpy as np
from pydantic import ValidatorFunctionWrapHandler, WrapValidator

mcp = FastMCP(name="similarity_tool")

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CACHE_DIR = "~/.cache/fastembed"
DEFAULT_MAX_TEXTS = 300
SCORE_DECIMALS = 4

TextSource = list[str] | str

# invariant: None in production; tests set a fake here.
_embedder: Any = None  # pylint: disable=invalid-name


def _model_name() -> str:
    """Return the EMBEDDING_MODEL value, or the default model when it is blank."""
    return os.environ.get("EMBEDDING_MODEL", "").strip() or DEFAULT_MODEL


def _max_texts() -> tuple[int, str | None]:
    """Read the text limit from SIMILARITY_MAX_TEXTS, where blank means unset.

    Returns:
        The limit and None, or 0 and the error message for an invalid value.
    """
    value = os.environ.get("SIMILARITY_MAX_TEXTS", "").strip()
    if not value:
        return DEFAULT_MAX_TEXTS, None
    invalid = 0, "SIMILARITY_MAX_TEXTS must be a positive integer"
    try:
        limit = int(value)
    except ValueError:
        return invalid
    return (limit, None) if limit > 0 else invalid


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


def _compact_json(payload: dict[str, Any]) -> str:
    """Serialize a response with no space after separators.

    Args:
        payload: The response object.

    Returns:
        The compact JSON text.
    """
    return json.dumps(payload, separators=(",", ":"))


def _keep_path(value: object, to_list: ValidatorFunctionWrapHandler) -> TextSource:
    """Let a path through, and validate anything else as a list of texts.

    Args:
        value: The raw argument.
        to_list: The pydantic validator of the list.

    Returns:
        The path, or the validated list.
    """
    return value if isinstance(value, str) else to_list(value)


# why: a plain union quotes the whole list in its str branch error, and a rejected
# call must not echo texts; this one keeps the list error and the union schema.
TextsArg = Annotated[
    list[str], WrapValidator(_keep_path, json_schema_input_type=TextSource)
]


def _absolute_path(path: str) -> str:
    """Expand "~" and anchor a relative path at the server working directory.

    Args:
        path: The path a client sent.

    Returns:
        The absolute path.
    """
    return os.path.abspath(os.path.expanduser(path))


def _load_texts(source: TextSource, name: str) -> list[str]:
    """Return the texts of an argument, reading a path as a JSON file.

    Args:
        source: The texts, or the path of a JSON file holding them.
        name: Name of the argument, for error messages.

    Returns:
        The texts.

    Raises:
        ValueError: If the path is blank or the file does not hold such an array.
    """
    if not isinstance(source, str):
        return source
    if not source.strip():
        raise ValueError(f"{name} is blank")
    path = _absolute_path(source)
    try:
        with open(path, encoding="utf-8") as file:
            texts = json.load(file)
    # why: ValueError covers bad JSON, bad UTF-8 and a NUL byte in the path, and a
    # too deep array raises RecursionError; each one must reach the client as data.
    except (OSError, ValueError, RecursionError) as exc:
        raise ValueError(f"cannot read {name}: {exc}") from exc
    if isinstance(texts, list) and all(isinstance(text, str) for text in texts):
        return texts
    raise ValueError(f"{name} must be a JSON array of strings: {path}")


def _validate(texts_a: list[str], texts_b: list[str] | None, limit: int) -> str | None:
    """Find the first problem in the input lists.

    Args:
        texts_a: Texts of the matrix rows.
        texts_b: Texts of the matrix columns, or None in one-list mode.
        limit: Highest text count across both lists.

    Returns:
        The error message, or None when the input is valid.
    """
    lists = {"texts_a": texts_a}
    if texts_b is not None:
        lists["texts_b"] = texts_b
    for name, texts in lists.items():
        if not texts:
            return f"{name} must not be empty"
    for name, texts in lists.items():
        for index, text in enumerate(texts):
            if not text.strip():
                return f"{name}[{index}] is blank"
    count = sum(len(texts) for texts in lists.values())
    return f"too many texts: {count} > {limit}" if count > limit else None


def _cosine_matrix(texts_a: list[str], texts_b: list[str] | None) -> dict[str, Any]:
    """Embed valid texts and score every pair by cosine similarity.

    Args:
        texts_a: Texts of the matrix rows.
        texts_b: Texts of the matrix columns, or None in one-list mode.

    Returns:
        {"matrix": [[...]], "model": "<name>"}, or {"error": "<message>"}.
    """
    texts = texts_a if texts_b is None else texts_a + texts_b
    model = _model_name()
    try:
        # hazard: embed may return a lazy generator, so list() must stay in this try.
        vectors = np.array(list(_get_embedder(model).embed(texts)), dtype=np.float64)
    # why: a model fails in many ways, and each one must reach the client as data.
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {"error": f"embedding failed: {exc}"}
    if len(vectors) != len(texts):
        count_problem = f"expected {len(texts)} vectors, got {len(vectors)}"
        return {"error": f"embedding failed: {count_problem}"}
    unit = _normalized(vectors)
    columns = unit if texts_b is None else unit[len(texts_a) :]
    matrix = np.round(unit[: len(texts_a)] @ columns.T, SCORE_DECIMALS).tolist()
    return {"matrix": matrix, "model": model}


def _scores(texts_a: TextSource, texts_b: TextSource | None) -> dict[str, Any]:
    """Load, validate and score the texts of a call.

    Args:
        texts_a: Texts of the matrix rows, or the path of a JSON file holding them.
        texts_b: The same for the matrix columns, or None in one-list mode.

    Returns:
        {"matrix": [[...]], "model": "<name>"}, or {"error": "<message>"}.
    """
    limit, problem = _max_texts()
    if problem:
        return {"error": problem}
    try:
        rows = _load_texts(texts_a, "texts_a")
        columns = None if texts_b is None else _load_texts(texts_b, "texts_b")
    except ValueError as exc:
        return {"error": str(exc)}
    problem = _validate(rows, columns, limit)
    return {"error": problem} if problem else _cosine_matrix(rows, columns)


@mcp.tool(structured_output=False)
def similarity_matrix(texts_a: TextsArg, texts_b: TextsArg | None = None) -> str:
    """Score every pair of texts by cosine similarity, from -1.0 to 1.0.

    A row or column of `matrix` is the position of a text in the input lists.
    Rows follow texts_a and columns follow texts_b.
    Omitting texts_b compares texts_a with itself.
    A string holding a JSON array is read as that list, and "null" as null.
    Any other string is the path of a JSON file holding an array of strings.
    A path has "~" expanded, and a relative one follows the server working directory.
    The default limit is 300 texts across both lists, counting those read from files.
    A bad call or an unreadable file returns {"error": "<message>"}.
    A text whose embedding has zero norm scores 0.0 on all its cells.

    Args:
        texts_a: Texts of the matrix rows, or the path of a JSON file holding them.
        texts_b: Texts of the matrix columns, or the path of a JSON file holding
            them; omit to compare texts_a with itself.

    Returns:
        Compact JSON text: {"matrix": [[...]], "model": "<name>"}, or {"error": ...}.
    """
    return _compact_json(_scores(texts_a, texts_b))


@mcp.tool(structured_output=False)
def similarity_matrix_csv(
    texts_a: TextsArg, output_path: str, texts_b: TextsArg | None = None
) -> str:
    """Save the similarity_matrix scores to a CSV file and return only its path.

    Takes the same texts, limits and rules as similarity_matrix.
    The CSV has no header: line i is texts_a[i], column j is texts_b[j].
    Omitting texts_b compares texts_a with itself.
    An existing file at output_path is overwritten.
    The parent directory must already exist.
    A bad call returns {"error": "<message>"} and writes nothing.

    Args:
        texts_a: Texts of the matrix rows, or the path of a JSON file holding them.
        output_path: CSV file to write; "~" expands, and a relative path follows
            the working directory of the server.
        texts_b: Texts of the matrix columns, or the path of a JSON file holding
            them; omit to compare texts_a with itself.

    Returns:
        Compact JSON text: {"path": "<absolute path>"}, or {"error": ...}.
    """
    if not output_path.strip():
        return _compact_json({"error": "output_path is blank"})
    result = _scores(texts_a, texts_b)
    if "error" in result:
        return _compact_json(result)
    path = _absolute_path(output_path)
    try:
        np.savetxt(path, result["matrix"], fmt=f"%.{SCORE_DECIMALS}f", delimiter=",")
    # why: ValueError covers a NUL byte in the path, which open rejects before the OS.
    except (OSError, ValueError) as exc:
        return _compact_json({"error": f"cannot write csv: {exc}"})
    return _compact_json({"path": path})


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
