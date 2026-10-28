"""Tests for the Clef MCP server, with the Cloudflare API mocked via httpx."""

# why: pytest fixtures are injected by parameter name, and tests are docstring-exempt.
# pylint: disable=redefined-outer-name,missing-function-docstring,unused-argument

import asyncio
import json
import pathlib
import tomllib

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


SINGLE_RESULT = {
    "model": "clef",
    "answers": {"answer": {"label": "b", "p": 0.8}},
    "usage": {"tokens": 7},
}


def single_ok(request):
    body = {"success": True, "errors": [], "result": SINGLE_RESULT}
    return httpx.Response(200, json=body)


def sent_questions(request):
    return json.loads(request.content)["questions"]


def test_choice_request(mock):
    seen = mock(single_ok)
    clef_mcp.clef_choice("s", "Pick one", {"a": "first", "b": "second"})
    assert sent_questions(seen[0]) == {
        "answer": {
            "type": "choice",
            "instructions": "Pick one",
            "criteria": {"a": "first", "b": "second"},
        }
    }


def test_score_request(mock):
    seen = mock(single_ok)
    clef_mcp.clef_score("s", "Rate it", ["low", "mid", "high"])
    assert sent_questions(seen[0]) == {
        "answer": {
            "type": "score",
            "instructions": "Rate it",
            "criteria": ["low", "mid", "high"],
        }
    }


def test_noul_request(mock):
    seen = mock(single_ok)
    clef_mcp.clef_noul("s", "Is it safe?")
    assert sent_questions(seen[0]) == {
        "answer": {"type": "noul", "instructions": "Is it safe?"}
    }


def test_single_answer_projection(mock):
    mock(single_ok)
    expected = {
        "answer": SINGLE_RESULT["answers"]["answer"],
        "model": "clef",
        "usage": {"tokens": 7},
    }
    assert clef_mcp.clef_choice("s", "i", {"a": "x", "b": "y"}) == expected
    assert clef_mcp.clef_score("s", "i", ["l", "h"]) == expected
    assert clef_mcp.clef_noul("s", "i") == expected


def test_single_error_passthrough(mock):
    body = {"success": False, "errors": [{"message": "quota gone"}]}
    mock(lambda r: httpx.Response(200, json=body))
    decide = clef_mcp.clef_decide("s", QUESTIONS)
    assert "quota gone" in decide["error"]
    assert clef_mcp.clef_choice("s", "i", {"a": "x", "b": "y"}) == decide
    assert clef_mcp.clef_score("s", "i", ["l", "h"]) == decide
    assert clef_mcp.clef_noul("s", "i") == decide


def test_single_model_argument(mock):
    seen = mock(single_ok)
    clef_mcp.clef_noul("s", "i", model="clef-flash")
    assert str(seen[0].url) == URL.format(model="clef-flash")
    out = clef_mcp.clef_choice("s", "i", {"a": "x", "b": "y"}, model="nope")
    assert "clef-flash" in out["error"]
    assert len(seen) == 1


@pytest.mark.parametrize(
    "result",
    [
        {"model": "clef", "usage": {}},
        {"model": "clef", "answers": "x", "usage": {}},
        {"model": "clef", "answers": {"other": {}}, "usage": {}},
    ],
)
def test_single_missing_answer(mock, result):
    mock(lambda r: httpx.Response(200, json={"success": True, "result": result}))
    out = clef_mcp.clef_noul("s", "i")
    assert "unexpected" in out["error"].lower()


def test_list_models(mock, monkeypatch):
    seen = mock(ok)
    assert clef_mcp.clef_list_models() == {
        "models": ["clef", "clef-flash"],
        "default": "clef",
    }
    monkeypatch.setenv("CLEF_MODEL", "clef-flash")
    assert clef_mcp.clef_list_models()["default"] == "clef-flash"
    assert seen == []


def test_tool_set():
    names = {tool.name for tool in asyncio.run(clef_mcp.mcp.list_tools())}
    assert names == {
        "clef_decide",
        "clef_choice",
        "clef_score",
        "clef_noul",
        "clef_list_models",
    }


def test_script_entry():
    root = pathlib.Path(__file__).resolve().parent.parent
    scripts = tomllib.loads((root / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["clef-mcp"] == "src.decision.clef_mcp:main"


def test_readme_documents_clef():
    root = pathlib.Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text()
    for item in (
        "clef-mcp",
        "clef_decide",
        "clef_choice",
        "clef_score",
        "clef_noul",
        "clef_list_models",
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "CLEF_MODEL",
    ):
        assert item in readme, item
