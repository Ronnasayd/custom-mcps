"""Tests for the similarity MCP server, with an injected embedder and a smoke tier."""

# why: pytest fixtures are injected by parameter name, and tests are docstring-exempt.
# pylint: disable=redefined-outer-name,missing-function-docstring,unused-argument
# pylint: disable=protected-access

import asyncio
import json
import os
import pathlib
import subprocess
import sys
import threading
import tomllib
import types

from mcp.shared.memory import create_connected_server_and_client_session
import numpy as np
import pytest

from src.embeddings import similarity_mcp

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORS = {
    "a": [1, 0],
    "b": [0, 1],
    "c": [-1, 0],
    "d": [3, 4],
    "e": [1, 2],
    "f": [3, 1],
    "zero": [0, 0],
    "  a \n": [1, 0],
}
SERVER = [sys.executable, "-m", "src.embeddings.similarity_mcp"]
OFFLINE = ["unshare", "-rn"]
STDIO_LIMIT_SECONDS = 120
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "smoke", "version": "1"},
    },
}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}
CALL = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
        "name": "similarity_matrix",
        "arguments": {"texts_a": ["gato", "cat", "carro"]},
    },
}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("SIMILARITY_MAX_TEXTS", raising=False)
    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
    similarity_mcp._load_embedder.cache_clear()


def vectors_for(texts):
    return (np.array(VECTORS[text], dtype=np.float32) for text in texts)


@pytest.fixture
def fake_embedder(monkeypatch):
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return vectors_for(texts)

    embedder = types.SimpleNamespace(embed=embed)
    monkeypatch.setattr(similarity_mcp, "_embedder", embedder)
    return calls


@pytest.fixture
def fake_fastembed(monkeypatch):
    built = []

    def text_embedding(model_name, cache_dir):
        built.append((model_name, cache_dir))
        return types.SimpleNamespace(embed=vectors_for)

    fastembed = types.SimpleNamespace(TextEmbedding=text_embedding)
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    return built


def similarity(*args):
    return json.loads(similarity_mcp.similarity_matrix(*args))


def call_over_wire(arguments):
    async def run():
        server = similarity_mcp.mcp
        async with create_connected_server_and_client_session(server) as session:
            return await session.call_tool("similarity_matrix", arguments)

    return asyncio.run(run())


def read_pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_script_entry():
    scripts = read_pyproject()["project"]["scripts"]
    assert scripts["similarity-mcp"] == "src.embeddings.similarity_mcp:main"


def test_embeddings_extra():
    project = read_pyproject()["project"]
    assert project["optional-dependencies"]["embeddings"] == ["fastembed==0.9.1"]
    assert not [dep for dep in project["dependencies"] if "fastembed" in dep]


def test_smoke_excluded_by_default():
    options = read_pyproject()["tool"]["pytest"]["ini_options"]
    assert options["addopts"] == "-m 'not smoke'"
    assert [line for line in options["markers"] if line.startswith("smoke:")]


def test_one_list_matrix(fake_embedder):
    assert similarity(["a", "b", "c", "d"])["matrix"] == [
        [1, 0, -1, 0.6],
        [0, 1, 0, 0.8],
        [-1, 0, 1, -0.6],
        [0.6, 0.8, -0.6, 1],
    ]


def test_response_fields(fake_embedder):
    assert set(similarity(["a", "b"])) == {"matrix", "model"}


def test_scores_rounded(fake_embedder):
    assert similarity(["e", "f"])["matrix"] == [[1.0, 0.7071], [0.7071, 1.0]]


def test_model_field(fake_embedder, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "org/custom-model")
    assert similarity(["a"])["model"] == "org/custom-model"


def test_model_blank_env(fake_embedder, monkeypatch):
    assert similarity(["a"])["model"] == DEFAULT_MODEL
    monkeypatch.setenv("EMBEDDING_MODEL", "   ")
    assert similarity(["a"])["model"] == DEFAULT_MODEL


def test_compact_json(fake_embedder):
    result = call_over_wire({"texts_a": ["e", "f"]})
    assert result.structuredContent is None
    assert result.content[0].text == (
        '{"matrix":[[1.0,0.7071],[0.7071,1.0]],"model":"' + DEFAULT_MODEL + '"}'
    )


def test_tool_set():
    names = {tool.name for tool in asyncio.run(similarity_mcp.mcp.list_tools())}
    assert names == {"similarity_matrix"}


def test_tool_description():
    description = asyncio.run(similarity_mcp.mcp.list_tools())[0].description
    assert "cosine similarity" in description
    assert "-1.0 to 1.0" in description
    assert "position of a text in the input lists" in description
    assert "Omitting texts_b compares texts_a with itself" in description
    assert "limit is 300" in description
    assert '{"error"' in description
    assert "zero norm scores 0.0" in description
    assert "first 128 tokens" in description


def test_zero_norm(fake_embedder):
    assert similarity(["zero", "a", "c"])["matrix"] == [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, -1.0],
        [0.0, -1.0, 1.0],
    ]


def test_single_text(fake_embedder):
    assert similarity(["a"])["matrix"] == [[1.0]]


def test_null_texts_b(fake_embedder):
    assert similarity(["a", "b"], None)["matrix"] == [[1.0, 0.0], [0.0, 1.0]]


def test_text_kept_unchanged(fake_embedder):
    similarity(["  a \n"])
    assert fake_embedder == [["  a \n"]]


def test_matrix_symmetric(fake_embedder):
    matrix = np.array(similarity(["a", "d", "e", "f"])["matrix"])
    assert np.abs(matrix - matrix.T).max() <= 0.0001


def test_loader_receives_env_model(fake_fastembed, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "org/custom-model")
    similarity(["a"])
    assert fake_fastembed[0][0] == "org/custom-model"


def test_one_embedder_per_model(fake_fastembed, monkeypatch, tmp_path):
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_MODEL", "org/one")
    similarity(["a"])
    similarity(["b"])
    assert fake_fastembed == [("org/one", str(tmp_path))]
    monkeypatch.setenv("EMBEDDING_MODEL", "org/two")
    similarity(["a"])
    assert fake_fastembed == [("org/one", str(tmp_path)), ("org/two", str(tmp_path))]


def test_cache_dir_default(fake_fastembed, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    expected = str(tmp_path / ".cache" / "fastembed")
    similarity_mcp._load_embedder("org/unset")
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", "   ")
    similarity_mcp._load_embedder("org/blank")
    assert fake_fastembed == [("org/unset", expected), ("org/blank", expected)]


def test_cache_dir_env(fake_fastembed, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", "/var/models")
    similarity_mcp._load_embedder("org/absolute")
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", "~/x")
    similarity_mcp._load_embedder("org/tilde")
    assert fake_fastembed == [
        ("org/absolute", "/var/models"),
        ("org/tilde", str(tmp_path / "x")),
    ]


def test_load_keeps_stdout_clean(monkeypatch, capsys):
    def noisy_text_embedding(model_name, cache_dir):
        print("downloading the model")

    fastembed = types.SimpleNamespace(TextEmbedding=noisy_text_embedding)
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    similarity_mcp._load_embedder("org/noisy")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "downloading the model" in captured.err


def test_import_without_fastembed():
    code = (
        "import sys\n"
        "sys.modules['fastembed'] = None\n"
        "import src.embeddings.similarity_mcp\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def run_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(similarity_mcp.mcp, "run", lambda: calls.append("run"))
    return calls


def test_main_loads_before_run(fake_fastembed, monkeypatch):
    loaded_at_run = []
    monkeypatch.setattr(
        similarity_mcp.mcp, "run", lambda: loaded_at_run.extend(fake_fastembed)
    )
    similarity_mcp.main()
    assert len(loaded_at_run) == 1
    assert loaded_at_run[0][0] == DEFAULT_MODEL


def test_main_ready_line(monkeypatch, capsys):
    def text_embedding(model_name, cache_dir):
        print("model loading", file=sys.stderr)

    fastembed = types.SimpleNamespace(TextEmbedding=text_embedding)
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    stderr_at_run = []
    monkeypatch.setattr(
        similarity_mcp.mcp, "run", lambda: stderr_at_run.append(capsys.readouterr().err)
    )
    similarity_mcp.main()
    assert stderr_at_run == ["model loading\nsimilarity-mcp: model ready\n"]


def test_main_missing_extra(run_calls, monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(SystemExit) as exit_info:
        similarity_mcp.main()
    message = exit_info.value.code
    assert isinstance(message, str)
    assert "--extra embeddings" in message
    assert "import of fastembed halted" in message
    assert run_calls == []


def test_main_load_failure(run_calls, monkeypatch):
    def text_embedding(model_name, cache_dir):
        raise OSError("disk is full")

    fastembed = types.SimpleNamespace(TextEmbedding=text_embedding)
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    with pytest.raises(SystemExit) as exit_info:
        similarity_mcp.main()
    message = exit_info.value.code
    assert isinstance(message, str)
    assert "disk is full" in message
    assert DEFAULT_MODEL in message
    assert run_calls == []


def send(process, *messages):
    for message in messages:
        process.stdin.write(json.dumps(message) + "\n")
    process.stdin.flush()


def talk_over_stdio(command, env):
    with subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as process:
        watchdog = threading.Timer(STDIO_LIMIT_SECONDS, process.kill)
        watchdog.start()
        try:
            send(process, INITIALIZE)
            lines = [process.stdout.readline()]
            send(process, INITIALIZED, CALL)
            lines.append(process.stdout.readline())
            process.stdin.close()
            lines.extend(process.stdout.readlines())
            process.wait()
        finally:
            watchdog.cancel()
    return [json.loads(line) for line in lines]


def assert_session_served(messages):
    initialize, call = messages
    assert initialize["jsonrpc"] == "2.0"
    assert initialize["id"] == 1
    assert initialize["result"]["serverInfo"]["name"] == "similarity_tool"
    assert call["jsonrpc"] == "2.0"
    assert call["id"] == 2
    assert call["result"]["isError"] is False
    body = json.loads(call["result"]["content"][0]["text"])
    assert body["model"] == DEFAULT_MODEL
    assert np.array(body["matrix"]).shape == (3, 3)


# invariant: this first smoke test fills the default cache the warm start reads.
@pytest.mark.smoke
def test_smoke_score_order():
    matrix = similarity(["gato", "cat", "carro"])["matrix"]
    assert matrix[0][1] > matrix[0][2]


@pytest.mark.smoke
def test_smoke_real_scores():
    matrix = np.array(similarity(["gato", "cat", "carro", "tax return"])["matrix"])
    assert matrix.min() >= -1.0
    assert matrix.max() <= 1.0
    assert np.diag(matrix).tolist() == [1.0, 1.0, 1.0, 1.0]


@pytest.mark.smoke
def test_smoke_long_text():
    out = similarity([" ".join(["palavra"] * 600), "curto"])
    assert "error" not in out
    assert np.array(out["matrix"]).shape == (2, 2)


@pytest.mark.smoke
def test_smoke_stdio_cold_cache(tmp_path):
    env = {**os.environ, "FASTEMBED_CACHE_PATH": str(tmp_path)}
    assert_session_served(talk_over_stdio(SERVER, env))


@pytest.mark.smoke
def test_smoke_offline_warm_start():
    assert_session_served(talk_over_stdio(OFFLINE + SERVER, dict(os.environ)))


@pytest.mark.smoke
def test_smoke_offline_empty_cache_fails(tmp_path):
    result = subprocess.run(
        OFFLINE + SERVER,
        cwd=ROOT,
        env={**os.environ, "FASTEMBED_CACHE_PATH": str(tmp_path)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=STDIO_LIMIT_SECONDS,
        check=False,
    )
    assert result.returncode != 0
    assert DEFAULT_MODEL in result.stderr


def test_two_list_matrix(fake_embedder):
    assert similarity(["a", "b"], ["a", "b", "c"])["matrix"] == [
        [1, 0, -1],
        [0, 1, 0],
    ]


def test_two_list_one_embed_call(fake_embedder):
    similarity(["a", "b"], ["a", "b", "c"])
    assert fake_embedder == [["a", "b", "a", "b", "c"]]


def test_empty_texts_a(fake_embedder):
    assert similarity([]) == {"error": "texts_a must not be empty"}
    assert fake_embedder == []


def test_empty_texts_b(fake_embedder):
    assert similarity(["a"], []) == {"error": "texts_b must not be empty"}
    assert fake_embedder == []


def test_blank_text(fake_embedder):
    assert similarity(["a", ""]) == {"error": "texts_a[1] is blank"}
    assert similarity(["a", "b", "c", " \t\n"]) == {"error": "texts_a[3] is blank"}
    assert similarity(["a"], ["b", "  "]) == {"error": "texts_b[1] is blank"}
    assert fake_embedder == []


def test_bad_types_is_error(fake_embedder):
    non_string_item = call_over_wire({"texts_a": ["a", 1]})
    number_as_list = call_over_wire({"texts_a": 5})
    number_as_texts_b = call_over_wire({"texts_a": ["a"], "texts_b": 5})
    missing_texts_a = call_over_wire({})
    assert non_string_item.isError is True
    assert "texts_a.1" in non_string_item.content[0].text
    assert number_as_list.isError is True
    assert "texts_a" in number_as_list.content[0].text
    assert number_as_texts_b.isError is True
    assert "texts_b" in number_as_texts_b.content[0].text
    assert missing_texts_a.isError is True
    assert "texts_a" in missing_texts_a.content[0].text
    assert fake_embedder == []


def test_too_many_texts(fake_embedder):
    assert similarity(["a"] * 150, ["b"] * 151) == {
        "error": "too many texts: 301 > 300"
    }
    assert similarity(["a"] * 301) == {"error": "too many texts: 301 > 300"}
    assert fake_embedder == []


def test_limit_exact(fake_embedder):
    two_lists = similarity(["a"] * 100, ["b"] * 200)
    one_list = similarity(["a"] * 300)
    assert np.array(two_lists["matrix"]).shape == (100, 200)
    assert np.array(one_list["matrix"]).shape == (300, 300)


def test_default_limit(fake_embedder, monkeypatch):
    too_many = {"error": "too many texts: 301 > 300"}
    assert similarity(["a"] * 301) == too_many
    assert np.array(similarity(["a"] * 300)["matrix"]).shape == (300, 300)
    monkeypatch.setenv("SIMILARITY_MAX_TEXTS", "   ")
    assert similarity(["a"] * 301) == too_many
    assert np.array(similarity(["a"] * 300)["matrix"]).shape == (300, 300)


def test_limit_from_env(fake_embedder, monkeypatch):
    monkeypatch.setenv("SIMILARITY_MAX_TEXTS", "2")
    assert similarity(["a", "b", "c"]) == {"error": "too many texts: 3 > 2"}
    assert similarity(["a", "b"], ["c"]) == {"error": "too many texts: 3 > 2"}
    assert similarity(["a"], ["c"])["matrix"] == [[-1.0]]
    assert similarity(["a", "b"])["matrix"] == [[1.0, 0.0], [0.0, 1.0]]


def test_invalid_limit(fake_embedder, monkeypatch):
    invalid = {"error": "SIMILARITY_MAX_TEXTS must be a positive integer"}
    monkeypatch.setenv("SIMILARITY_MAX_TEXTS", "many")
    assert similarity(["a"]) == invalid
    monkeypatch.setenv("SIMILARITY_MAX_TEXTS", "0")
    assert similarity(["a"]) == invalid
    monkeypatch.setenv("SIMILARITY_MAX_TEXTS", "-5")
    assert similarity(["a"]) == invalid
    assert fake_embedder == []


def test_first_problem_reported(fake_embedder, monkeypatch):
    assert similarity(["a"] * 301 + [" "]) == {"error": "texts_a[301] is blank"}
    assert similarity([" "], []) == {"error": "texts_b must not be empty"}
    monkeypatch.setenv("SIMILARITY_MAX_TEXTS", "none")
    assert similarity([]) == {
        "error": "SIMILARITY_MAX_TEXTS must be a positive integer"
    }
