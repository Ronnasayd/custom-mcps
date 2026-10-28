"""FastMCP server exposing Cloudflare's hosted Clef decision models as typed tools."""

import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="clef_tool")

MODELS = ("clef", "clef-flash")
DEFAULT_MODEL = "clef"
API_URL = (
    "https://api.cloudflare.com/client/v4/accounts/{account}"
    "/ai/run/@cf/cloudflare/{model}"
)
TIMEOUT_SECONDS = 30

# invariant: tests inject an httpx.MockTransport here; None means the real network.
_transport: httpx.BaseTransport | None = None  # pylint: disable=invalid-name

State = str | dict[str, Any] | list
type JsonValue = (
    str | int | float | bool | None | dict[str, JsonValue] | list[JsonValue]
)


def _default_model() -> str:
    """Return the CLEF_MODEL value, or the built-in default when it is blank."""
    return os.environ.get("CLEF_MODEL", "").strip() or DEFAULT_MODEL


def _resolve_model(model: str | None) -> tuple[str, str | None]:
    """Normalize and validate a model name.

    Args:
        model: Explicit model argument, or None to use the default.

    Returns:
        A (model, error) pair; error is None when the model is allowed.
    """
    name = (model if model is not None else _default_model()).strip().lower()
    if name not in MODELS:
        return name, f"Invalid model {name!r}. Allowed models: {', '.join(MODELS)}"
    return name, None


def _credentials() -> tuple[str, str, str | None]:
    """Read the Cloudflare credentials from the environment at call time.

    Returns:
        An (account_id, token, error) triple; error names the first missing variable.
    """
    values = []
    for var in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"):
        value = os.environ.get(var, "").strip()
        if not value:
            return "", "", f"Missing environment variable {var}"
        values.append(value)
    return values[0], values[1], None


def _error_messages(body: dict[str, Any], status: int) -> str:
    """Join the Cloudflare envelope's error messages into one string.

    Args:
        body: Parsed JSON envelope with ``success`` false.
        status: HTTP status code of the response.

    Returns:
        The messages joined by "; ", or a generic failure line when there are none.
    """
    errors = body.get("errors")
    if not isinstance(errors, list):
        errors = []
    messages = [e["message"] for e in errors if isinstance(e, dict) and "message" in e]
    return "; ".join(messages) or f"API reported failure (HTTP {status})"


def _send(
    account: str, token: str, model: str, state: State, questions: dict[str, Any]
) -> httpx.Response | dict[str, Any]:
    """POST one request to Cloudflare.

    Args:
        account: Cloudflare account id.
        token: Cloudflare API token.
        model: Validated model name.
        state: Context the questions are asked about.
        questions: Map of question id to typed question.

    Returns:
        The HTTP response, or ``{"error": ...}`` on a transport failure.
    """
    url = API_URL.format(account=account, model=model)
    headers = {"Authorization": f"Bearer {token}"}
    body = {"model": model, "state": state, "questions": questions}
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS, transport=_transport) as client:
            return client.post(url, headers=headers, json=body)
    except httpx.TimeoutException:
        return {"error": f"Request timed out after {TIMEOUT_SECONDS}s"}
    except httpx.ConnectError as exc:
        return {"error": f"Could not connect to Cloudflare: {exc}"}
    except httpx.HTTPError as exc:
        return {"error": f"Request failed: {exc}"}


def _parse(response: httpx.Response) -> dict[str, Any]:
    """Turn a Cloudflare response into its ``result`` object or an error dict.

    Args:
        response: HTTP response from the Workers AI REST API.

    Returns:
        The envelope's ``result`` object, or ``{"error": ...}``.
    """
    status = response.status_code
    try:
        payload = response.json()
    except ValueError:
        return {"error": f"Non-JSON response (HTTP {status})"}
    if not isinstance(payload, dict) or "success" not in payload:
        return {"error": f"Unexpected response shape (HTTP {status})"}
    if payload["success"] is not True:
        return {"error": _error_messages(payload, status)}
    if not isinstance(payload.get("result"), dict):
        return {"error": f"Unexpected response shape: no result (HTTP {status})"}
    return payload["result"]


def _scrub(value: JsonValue, token: str) -> JsonValue:
    """Replace every occurrence of the token in strings nested inside a value.

    Args:
        value: Any JSON-like value (str, dict, list or scalar).
        token: Secret to remove.

    Returns:
        A copy of the value with the token replaced by ``[redacted]``.
    """
    if isinstance(value, str):
        return value.replace(token, "[redacted]")
    if isinstance(value, dict):
        return {k: _scrub(v, token) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v, token) for v in value]
    return value


def _post(model: str | None, state: State, questions: dict[str, Any]) -> dict[str, Any]:
    """Validate inputs, call the API and scrub the token from any error.

    Args:
        model: Explicit model name, or None for the default.
        state: Context the questions are asked about.
        questions: Map of question id to typed question.

    Returns:
        The API ``result`` object, or ``{"error": ...}``.
    """
    name, error = _resolve_model(model)
    if error:
        return {"error": error}
    account, token, error = _credentials()
    if error:
        return {"error": error}
    sent = _send(account, token, name, state, questions)
    out = sent if isinstance(sent, dict) else _parse(sent)
    return {key: _scrub(value, token) for key, value in out.items()}


@mcp.tool()
def clef_decide(
    state: State, questions: dict[str, Any], model: str | None = None
) -> dict[str, Any]:
    """Ask Clef several typed questions about a state in one call.

    Args:
        state: Context the questions are asked about (text, object or list).
        questions: Map of question id to ``{"type": "choice"|"score"|"noul", ...}``.
        model: ``clef`` or ``clef-flash``; defaults to CLEF_MODEL, then ``clef``.

    Returns:
        The API ``result`` (answers, model, usage), or ``{"error": ...}``.
    """
    return _post(model, state, questions)


def _ask_one(
    model: str | None, state: State, question: dict[str, Any]
) -> dict[str, Any]:
    """Send one question under the id ``answer`` and project the reply.

    Args:
        model: Explicit model name, or None for the default.
        state: Context the question is asked about.
        question: Typed question body.

    Returns:
        ``{"answer", "model", "usage"}`` on success, or ``{"error": ...}``.
    """
    out = _post(model, state, {"answer": question})
    if "error" in out:
        return out
    answers = out.get("answers")
    if not isinstance(answers, dict) or "answer" not in answers:
        return {"error": "Unexpected response shape: result has no answers.answer"}
    return {
        "answer": answers["answer"],
        "model": out.get("model"),
        "usage": out.get("usage"),
    }


@mcp.tool()
def clef_choice(
    state: State, instructions: str, options: dict[str, str], model: str | None = None
) -> dict[str, Any]:
    """Pick one option from a labelled set.

    Args:
        state: Context the question is asked about.
        instructions: What to decide.
        options: Map of option label to its description (2-255 options).
        model: ``clef`` or ``clef-flash``; defaults to CLEF_MODEL, then ``clef``.

    Returns:
        ``{"answer", "model", "usage"}``, or ``{"error": ...}``.
    """
    question = {"type": "choice", "instructions": instructions, "criteria": options}
    return _ask_one(model, state, question)


@mcp.tool()
def clef_score(
    state: State, instructions: str, levels: list[str], model: str | None = None
) -> dict[str, Any]:
    """Rate the state on an ordered scale.

    Args:
        state: Context the question is asked about.
        instructions: What to rate.
        levels: Ordered level descriptions (2-10 levels).
        model: ``clef`` or ``clef-flash``; defaults to CLEF_MODEL, then ``clef``.

    Returns:
        ``{"answer", "model", "usage"}``, or ``{"error": ...}``.
    """
    question = {"type": "score", "instructions": instructions, "criteria": levels}
    return _ask_one(model, state, question)


@mcp.tool()
def clef_noul(
    state: State, instructions: str, model: str | None = None
) -> dict[str, Any]:
    """Estimate the probability that a yes/no statement holds.

    Args:
        state: Context the question is asked about.
        instructions: The statement to judge.
        model: ``clef`` or ``clef-flash``; defaults to CLEF_MODEL, then ``clef``.

    Returns:
        ``{"answer", "model", "usage"}``, or ``{"error": ...}``.
    """
    return _ask_one(model, state, {"type": "noul", "instructions": instructions})


@mcp.tool()
def clef_list_models() -> dict[str, Any]:
    """List the supported Clef models and the default used when none is passed.

    Returns:
        ``{"models": [...], "default": ...}``; no network request is made.
    """
    return {"models": list(MODELS), "default": _default_model().lower()}


def main() -> None:
    """Run the Clef MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
