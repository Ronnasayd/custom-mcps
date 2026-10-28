"""Tests for the Clef MCP server, with the Cloudflare API mocked via httpx."""

# why: pytest fixtures are injected by parameter name, and tests are docstring-exempt.
# pylint: disable=redefined-outer-name,missing-function-docstring,unused-argument

import json

import httpx
import pytest

from src.decision import clef_mcp

ACCOUNT = "acc123"
TOKEN = "tok-secret-value"
URL = (
    "https://api.cloudflare.com/client/v4/accounts/acc123/ai/run/@cf/cloudflare/{model}"
)
QUESTIONS = {"q1": {"type": "noul", "instructions": "Is it safe?"}}
RESULT = {"model": "clef", "answers": {"q1": {"p": 0.9}}, "usage": {"tokens": 5}}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", TOKEN)
    monkeypatch.delenv("CLEF_MODEL", raising=False)


@pytest.fixture
def mock(monkeypatch):

    def install(handler):
        seen = []

        def wrapper(request):
            seen.append(request)
            return handler(request)

        monkeypatch.setattr(clef_mcp, "_transport", httpx.MockTransport(wrapper))
        return seen

    return install


def ok(request):
    return httpx.Response(200, json={"success": True, "errors": [], "result": RESULT})


def test_decide_request_shape(mock):
    seen = mock(ok)
    clef_mcp.clef_decide({"a": 1}, QUESTIONS)
    req = seen[0]
    assert req.method == "POST"
    assert str(req.url) == URL.format(model="clef")
    assert req.headers["authorization"] == f"Bearer {TOKEN}"
    assert json.loads(req.content) == {
        "model": "clef",
        "state": {"a": 1},
        "questions": QUESTIONS,
    }


def test_decide_returns_result(mock):
    mock(ok)
    assert clef_mcp.clef_decide("s", QUESTIONS) == RESULT


@pytest.mark.parametrize("env_value", [None, "", "   "])
def test_default_model(mock, monkeypatch, env_value):
    seen = mock(ok)
    if env_value is not None:
        monkeypatch.setenv("CLEF_MODEL", env_value)
    clef_mcp.clef_decide("s", QUESTIONS)
    assert str(seen[0].url) == URL.format(model="clef")
    monkeypatch.setenv("CLEF_MODEL", "clef-flash")
    clef_mcp.clef_decide("s", QUESTIONS)
    assert str(seen[1].url) == URL.format(model="clef-flash")


def test_clef_flash_routing(mock):
    seen = mock(ok)
    clef_mcp.clef_decide("s", QUESTIONS, model="clef-flash")
    assert str(seen[0].url) == URL.format(model="clef-flash")
    assert json.loads(seen[0].content)["model"] == "clef-flash"


def test_invalid_model(mock):
    seen = mock(ok)
    out = clef_mcp.clef_decide("s", QUESTIONS, model="gpt-4")
    assert set(out) == {"error"}
    assert "clef" in out["error"] and "clef-flash" in out["error"]
    assert seen == []
    clef_mcp.clef_decide("s", QUESTIONS, model="  CLEF-Flash ")
    assert str(seen[0].url) == URL.format(model="clef-flash")


def test_invalid_model_env(mock, monkeypatch):
    seen = mock(ok)
    monkeypatch.setenv("CLEF_MODEL", "bogus")
    out = clef_mcp.clef_decide("s", QUESTIONS)
    assert "clef-flash" in out["error"]
    assert seen == []


@pytest.mark.parametrize("var", ["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"])
@pytest.mark.parametrize("value", [None, "", "  "])
def test_missing_credentials(mock, monkeypatch, var, value):
    seen = mock(ok)
    if value is None:
        monkeypatch.delenv(var)
    else:
        monkeypatch.setenv(var, value)
    out = clef_mcp.clef_decide("s", QUESTIONS)
    assert var in out["error"]
    assert seen == []


def test_api_error_envelope(mock):
    body = {
        "success": False,
        "errors": [{"message": "bad one"}, {"message": "bad two"}],
    }
    mock(lambda r: httpx.Response(200, json=body))
    out = clef_mcp.clef_decide("s", QUESTIONS)
    assert "bad one" in out["error"] and "bad two" in out["error"]


def test_api_error_http_status(mock):
    body = {"success": False, "errors": [{"message": "limit hit"}]}
    mock(lambda r: httpx.Response(429, json=body))
    assert "limit hit" in clef_mcp.clef_decide("s", QUESTIONS)["error"]


def _raise(exc):
    def handler(request):
        raise exc

    return handler


@pytest.mark.parametrize(
    "handler, fragment",
    [
        (_raise(httpx.ReadTimeout("slow")), "timed out"),
        (_raise(httpx.ConnectError("refused")), "connect"),
        (lambda r: httpx.Response(502, text="<html>bad gateway</html>"), "502"),
        (lambda r: httpx.Response(200, text="not json"), "JSON"),
    ],
)
def test_transport_failure(mock, handler, fragment):
    mock(handler)
    out = clef_mcp.clef_decide("s", QUESTIONS)
    assert set(out) == {"error"}
    assert fragment.lower() in out["error"].lower()


def test_token_never_leaked(mock):
    echo = {"success": False, "errors": [{"message": f"bad token {TOKEN}"}]}
    cases = [
        lambda r: httpx.Response(200, json=echo),
        _raise(httpx.ConnectError(f"cannot reach with {TOKEN}")),
        lambda r: httpx.Response(500, text=f"oops {TOKEN}"),
    ]
    for handler in cases:
        mock(handler)
        out = clef_mcp.clef_decide("s", QUESTIONS)
        assert "error" in out
        assert TOKEN not in json.dumps(out)


@pytest.mark.parametrize(
    "payload",
    [[1, 2], {"errors": []}, {"success": True}, {"success": True, "result": "x"}],
)
def test_unexpected_shape(mock, payload):
    mock(lambda r: httpx.Response(200, json=payload))
    out = clef_mcp.clef_decide("s", QUESTIONS)
    assert "unexpected" in out["error"].lower()
