"""Tests for the similarity MCP server, with an injected embedder and a smoke tier."""

# why: pytest fixtures are injected by parameter name, and tests are docstring-exempt.
# pylint: disable=redefined-outer-name,missing-function-docstring,unused-argument
# pylint: disable=protected-access
# why: one suite per server; a split would only move the shared fixtures around.
# pylint: disable=too-many-lines

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


def similarity_csv(*args):
    return json.loads(similarity_mcp.similarity_matrix_csv(*args))


def call_over_wire(arguments, tool="similarity_matrix"):
    async def run():
        server = similarity_mcp.mcp
        async with create_connected_server_and_client_session(server) as session:
            return await session.call_tool(tool, arguments)

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
    assert names == {"similarity_matrix", "similarity_matrix_csv"}


def test_tool_description():
    description = asyncio.run(similarity_mcp.mcp.list_tools())[0].description
    assert "cosine similarity" in description
    assert "-1.0 to 1.0" in description
    assert "position of a text in the input lists" in description
    assert "Omitting texts_b compares texts_a with itself" in description
    assert "limit is 300" in description
    assert '{"error"' in description
    assert "zero norm scores 0.0" in description
    assert "128 tokens" not in description


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


def use_embedder(monkeypatch, embed):
    embedder = types.SimpleNamespace(embed=embed)
    monkeypatch.setattr(similarity_mcp, "_embedder", embedder)


def broken_stream():
    yield np.array([1.0, 0.0])
    raise RuntimeError("stream broke")


def scripted_embed(texts):
    if "boom" in texts:
        raise RuntimeError("model crashed")
    if "mid" in texts:
        return broken_stream()
    if "short" in texts:
        return vectors_for(["a"])
    return vectors_for(texts)


def in_session(scenario):
    async def run():
        server = similarity_mcp.mcp
        async with create_connected_server_and_client_session(server) as session:
            await scenario(session)

    asyncio.run(run())


async def call_tool(session, arguments):
    return await session.call_tool("similarity_matrix", arguments)


async def assert_serves(session):
    result = await call_tool(session, {"texts_a": ["a", "c"]})
    assert result.isError is False
    assert json.loads(result.content[0].text)["matrix"] == [[1.0, -1.0], [-1.0, 1.0]]


async def assert_error_data(session, arguments, message):
    result = await call_tool(session, arguments)
    assert result.isError is False
    assert json.loads(result.content[0].text) == {"error": message}


async def assert_rejected(session, arguments, field):
    result = await call_tool(session, arguments)
    assert result.isError is True
    assert field in result.content[0].text


def test_embed_raises(monkeypatch):
    use_embedder(monkeypatch, scripted_embed)
    assert similarity(["boom"]) == {"error": "embedding failed: model crashed"}


def test_embed_raises_mid_iteration(monkeypatch):
    use_embedder(monkeypatch, scripted_embed)
    assert similarity(["mid"]) == {"error": "embedding failed: stream broke"}


def test_load_raises_on_call(monkeypatch):
    def text_embedding(model_name, cache_dir):
        raise OSError("disk is full")

    fastembed = types.SimpleNamespace(TextEmbedding=text_embedding)
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    assert similarity(["a"]) == {"error": "embedding failed: disk is full"}


def test_vector_count_mismatch(monkeypatch):
    use_embedder(monkeypatch, lambda texts: vectors_for(["a", "b"]))
    assert similarity(["a", "b", "c"]) == {
        "error": "embedding failed: expected 3 vectors, got 2"
    }


def test_next_call_after_errors(monkeypatch):
    def text_embedding(model_name, cache_dir):
        if model_name == "org/broken":
            raise OSError("disk is full")
        return types.SimpleNamespace(embed=scripted_embed)

    async def scenario(session):
        await assert_error_data(session, {"texts_a": []}, "texts_a must not be empty")
        await assert_serves(session)
        await assert_error_data(
            session, {"texts_a": ["a"], "texts_b": []}, "texts_b must not be empty"
        )
        await assert_serves(session)
        await assert_error_data(session, {"texts_a": [" "]}, "texts_a[0] is blank")
        await assert_serves(session)
        await assert_error_data(
            session, {"texts_a": ["a"] * 301}, "too many texts: 301 > 300"
        )
        await assert_serves(session)
        monkeypatch.setenv("SIMILARITY_MAX_TEXTS", "none")
        await assert_error_data(
            session,
            {"texts_a": ["a"]},
            "SIMILARITY_MAX_TEXTS must be a positive integer",
        )
        monkeypatch.delenv("SIMILARITY_MAX_TEXTS")
        await assert_serves(session)
        await assert_rejected(session, {"texts_a": ["a", 1]}, "texts_a.1")
        await assert_serves(session)
        await assert_rejected(session, {"texts_a": 5}, "texts_a")
        await assert_serves(session)
        await assert_rejected(session, {}, "texts_a")
        await assert_serves(session)
        await assert_error_data(
            session, {"texts_a": ["boom"]}, "embedding failed: model crashed"
        )
        await assert_serves(session)
        await assert_error_data(
            session, {"texts_a": ["mid"]}, "embedding failed: stream broke"
        )
        await assert_serves(session)
        await assert_error_data(
            session,
            {"texts_a": ["short", "a", "b"]},
            "embedding failed: expected 3 vectors, got 1",
        )
        await assert_serves(session)
        monkeypatch.setenv("EMBEDDING_MODEL", "org/broken")
        await assert_error_data(
            session, {"texts_a": ["a"]}, "embedding failed: disk is full"
        )
        monkeypatch.delenv("EMBEDDING_MODEL")
        await assert_serves(session)

    fastembed = types.SimpleNamespace(TextEmbedding=text_embedding)
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    in_session(scenario)


def test_no_text_logged(monkeypatch, capfd, caplog):
    sentinel = "sentinel-input-text"

    def constant_embed(texts):
        return (np.array([1.0, 0.0]) for _ in texts)

    def quoting_embed(texts):
        raise RuntimeError(f"cannot read {texts}")

    # invariant: the level the server itself configures, so records match production.
    caplog.set_level(similarity_mcp.mcp.settings.log_level)
    use_embedder(monkeypatch, constant_embed)
    success = call_over_wire({"texts_a": [sentinel]})
    blank = call_over_wire({"texts_a": [sentinel, " "]})
    rejected = call_over_wire({"texts_a": [sentinel, 1]})
    use_embedder(monkeypatch, scripted_embed)
    failed = call_over_wire({"texts_a": [sentinel, "boom"]})
    use_embedder(monkeypatch, quoting_embed)
    call_over_wire({"texts_a": [sentinel]})
    captured = capfd.readouterr()
    assert "Processing request of type CallToolRequest" in caplog.text
    assert sentinel not in caplog.text
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert json.loads(success.content[0].text)["matrix"] == [[1.0]]
    assert blank.content[0].text == '{"error":"texts_a[1] is blank"}'
    assert sentinel not in rejected.content[0].text
    assert failed.content[0].text == '{"error":"embedding failed: model crashed"}'


def read_readme():
    return (ROOT / "README.md").read_text()


def readme_section(start, end="\n## "):
    return read_readme().split(start, 1)[1].split(end, 1)[0]


def test_readme_row():
    lines = read_readme().splitlines()
    row = next(line for line in lines if line.startswith("| `similarity-mcp`"))
    _, command, tool, database, variables, _ = row.split("|")
    assert command.strip() == "`similarity-mcp`"
    assert tool.strip() == "`similarity_matrix`, `similarity_matrix_csv`"
    assert database.strip() == "-"
    assert "`EMBEDDING_MODEL`" in variables
    assert "`SIMILARITY_MAX_TEXTS`" in variables
    assert "`FASTEMBED_CACHE_PATH`" in variables


def test_readme_section():
    section = readme_section("### `similarity-mcp`")
    assert "similarity_matrix(texts_a, texts_b=None)" in section
    assert "One list: omit `texts_b`" in section
    assert "Two lists: pass `texts_b`" in section
    assert "`matrix`" in section
    assert "`model`" in section
    assert "300 texts" in section
    assert "`SIMILARITY_MAX_TEXTS`" in section
    assert "first 128 tokens" in section
    assert "241 MB" in section
    assert "exits with a non-zero status" in section
    assert (
        "--directory /path/to/custom-mcps --extra embeddings similarity-mcp" in section
    )
    assert "`EMBEDDING_MODEL`" in section
    assert "`FASTEMBED_CACHE_PATH`" in section
    assert "`similarity-mcp: model ready`" in section
    # why: SPEC_DEVIATION from design.md, measured on a pty: Ctrl+C leaves the
    # server blocked on stdin, and Ctrl+D ends it with status 0.
    assert "stop it with Ctrl+D" in section
    assert "then register" in section


def test_readme_layout():
    assert "src/embeddings/\n  similarity_mcp.py\n" in readme_section("## Layout")


def test_readme_security():
    section = readme_section("## Security", "\n## Layout")
    assert (
        "`similarity-mcp` touches no database and contacts the network only "
        "to download the model" in section
    )


def test_readme_opening():
    opening = read_readme().split("\n\n")[1]
    assert "text similarity server" in opening


def test_limit_with_plus_sign(fake_embedder, monkeypatch):
    monkeypatch.setenv("SIMILARITY_MAX_TEXTS", "+2")
    assert similarity(["a", "b"])["matrix"] == [[1.0, 0.0], [0.0, 1.0]]
    assert similarity(["a", "b", "c"]) == {"error": "too many texts: 3 > 2"}


def test_limit_too_long_to_parse(fake_embedder, monkeypatch):
    monkeypatch.setenv("SIMILARITY_MAX_TEXTS", "1" + "0" * 5000)
    assert similarity(["a"]) == {
        "error": "SIMILARITY_MAX_TEXTS must be a positive integer"
    }
    assert fake_embedder == []


def test_json_text_arguments(fake_embedder):
    identity = [[1.0, 0.0], [0.0, 1.0]]
    array_text = call_over_wire({"texts_a": '["a", "b"]'})
    null_text = call_over_wire({"texts_a": ["a", "b"], "texts_b": "null"})
    json_null = call_over_wire({"texts_a": ["a", "b"], "texts_b": None})
    for result in (array_text, null_text, json_null):
        assert result.isError is False
        assert json.loads(result.content[0].text)["matrix"] == identity
    assert fake_embedder == [["a", "b"]] * 3


def test_json_text_documented():
    description = asyncio.run(similarity_mcp.mcp.list_tools())[0].description
    section = readme_section("### `similarity-mcp`", "\n### ")
    assert "string holding a JSON array" in description
    assert "string holding a JSON array" in section
    assert "`null`" in section


def test_csv_two_lists(fake_embedder, tmp_path):
    path = tmp_path / "scores.csv"
    assert similarity_csv(["a", "b"], str(path), ["c", "d"]) == {"path": str(path)}
    assert path.read_text() == "-1.0000,0.6000\n0.0000,0.8000\n"


def test_csv_one_list(fake_embedder, tmp_path):
    path = tmp_path / "scores.csv"
    similarity_csv(["a", "b"], str(path))
    assert path.read_text() == "1.0000,0.0000\n0.0000,1.0000\n"
    assert fake_embedder == [["a", "b"]]


def test_csv_same_scores_as_matrix(fake_embedder, tmp_path):
    path = tmp_path / "scores.csv"
    similarity_csv(["a", "d", "e"], str(path), ["b", "f"])
    expected = similarity(["a", "d", "e"], ["b", "f"])["matrix"]
    assert np.loadtxt(path, delimiter=",").tolist() == expected


def test_csv_overwrites(fake_embedder, tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("old")
    similarity_csv(["a"], str(path))
    assert path.read_text() == "1.0000\n"


def test_csv_relative_path_returned_absolute(fake_embedder, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert similarity_csv(["a"], "scores.csv") == {"path": str(tmp_path / "scores.csv")}
    assert (tmp_path / "scores.csv").exists()


def test_csv_expands_home(fake_embedder, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert similarity_csv(["a"], "~/scores.csv") == {
        "path": str(tmp_path / "scores.csv")
    }


@pytest.mark.parametrize("blank", ["", "   "])
def test_csv_blank_path(fake_embedder, blank):
    assert similarity_csv(["a"], blank) == {"error": "output_path is blank"}
    assert fake_embedder == []


def test_csv_bad_input_writes_nothing(fake_embedder, tmp_path):
    path = tmp_path / "scores.csv"
    assert similarity_csv([], str(path)) == {"error": "texts_a must not be empty"}
    assert not path.exists()


def test_csv_embedding_failure_writes_nothing(tmp_path, monkeypatch):
    def embed(texts):
        raise RuntimeError("model crashed")

    embedder = types.SimpleNamespace(embed=embed)
    monkeypatch.setattr(similarity_mcp, "_embedder", embedder)
    path = tmp_path / "scores.csv"
    assert similarity_csv(["a"], str(path)) == {
        "error": "embedding failed: model crashed"
    }
    assert not path.exists()


@pytest.mark.parametrize("name", ["missing/scores.csv", ".", "bad\0name.csv"])
def test_csv_unwritable_path(fake_embedder, tmp_path, name):
    out = similarity_csv(["a"], f"{tmp_path}/{name}")
    assert out["error"].startswith("cannot write csv: ")


def test_csv_over_wire(fake_embedder, tmp_path):
    path = tmp_path / "scores.csv"
    result = call_over_wire(
        {"texts_a": '["a", "b"]', "output_path": str(path), "texts_b": "null"},
        "similarity_matrix_csv",
    )
    assert result.isError is False
    assert json.loads(result.content[0].text) == {"path": str(path)}
    assert path.read_text() == "1.0000,0.0000\n0.0000,1.0000\n"


def test_csv_documented():
    section = readme_section("### `similarity-mcp`", "\n### ")
    assert "similarity_matrix_csv(texts_a, output_path, texts_b=None)" in section
    assert '{"path":"/tmp/scores.csv"}' in section
    assert "overwrites" in section


@pytest.mark.smoke
def test_smoke_csv(tmp_path):
    path = tmp_path / "scores.csv"
    similarity_csv(["gato", "cat", "carro"], str(path))
    scores = np.loadtxt(path, delimiter=",")
    assert scores.shape == (3, 3)
    assert scores[0][1] > scores[0][2]


def write_texts(path, texts):
    path.write_text(json.dumps(texts), encoding="utf-8")
    return str(path)


def test_list_input_unchanged(fake_embedder):
    result = call_over_wire({"texts_a": ["a", "b"], "texts_b": ["c", "d"]})
    assert json.loads(result.content[0].text)["matrix"] == [[-1.0, 0.6], [0.0, 0.8]]
    assert fake_embedder == [["a", "b", "c", "d"]]


def test_path_in_texts_a(fake_embedder, tmp_path):
    path = write_texts(tmp_path / "a.json", ["a", "b"])
    assert similarity(path) == similarity(["a", "b"])
    assert fake_embedder == [["a", "b"]] * 2


def test_path_in_texts_b(fake_embedder, tmp_path):
    path = write_texts(tmp_path / "b.json", ["c", "d"])
    assert similarity(["a", "b"], path)["matrix"] == [[-1.0, 0.6], [0.0, 0.8]]


def test_path_in_both(fake_embedder, tmp_path):
    rows = write_texts(tmp_path / "a.json", ["a", "b"])
    columns = write_texts(tmp_path / "b.json", ["c", "d"])
    assert similarity(rows, columns)["matrix"] == [[-1.0, 0.6], [0.0, 0.8]]
    assert fake_embedder == [["a", "b", "c", "d"]]


def test_path_mixed_with_list(fake_embedder, tmp_path):
    path = write_texts(tmp_path / "texts.json", ["c", "d"])
    assert similarity(path, ["a", "b"])["matrix"] == [[-1.0, 0.0], [0.6, 0.8]]
    assert fake_embedder == [["c", "d", "a", "b"]]


def test_path_expands_home(fake_embedder, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    write_texts(tmp_path / "texts.json", ["a", "b"])
    assert similarity("~/texts.json")["matrix"] == [[1.0, 0.0], [0.0, 1.0]]


def test_path_relative_follows_cwd(fake_embedder, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_texts(tmp_path / "texts.json", ["a", "b"])
    assert similarity("texts.json")["matrix"] == [[1.0, 0.0], [0.0, 1.0]]


def test_path_read_as_utf8(tmp_path, monkeypatch):
    seen = []

    def embed(texts):
        seen.extend(texts)
        return (np.array([1.0, 0.0]) for _ in texts)

    use_embedder(monkeypatch, embed)
    path = tmp_path / "texts.json"
    path.write_bytes('["olá"]'.encode())
    assert similarity(str(path))["matrix"] == [[1.0]]
    assert seen == ["olá"]


def test_path_over_wire(fake_embedder, tmp_path):
    rows = write_texts(tmp_path / "a.json", ["a", "b"])
    result = call_over_wire({"texts_a": rows, "texts_b": ["c", "d"]})
    assert result.isError is False
    assert json.loads(result.content[0].text)["matrix"] == [[-1.0, 0.6], [0.0, 0.8]]


def test_csv_with_path(fake_embedder, tmp_path):
    rows = write_texts(tmp_path / "a.json", ["a", "b"])
    out = tmp_path / "scores.csv"
    result = call_over_wire(
        {"texts_a": rows, "output_path": str(out), "texts_b": ["c", "d"]},
        "similarity_matrix_csv",
    )
    assert json.loads(result.content[0].text) == {"path": str(out)}
    assert out.read_text() == "-1.0000,0.6000\n0.0000,0.8000\n"


def test_schema_offers_list_or_path():
    list_or_path = [{"items": {"type": "string"}, "type": "array"}, {"type": "string"}]
    for tool in asyncio.run(similarity_mcp.mcp.list_tools()):
        properties = tool.inputSchema["properties"]
        assert properties["texts_a"]["anyOf"] == list_or_path
        assert properties["texts_b"]["anyOf"] == [*list_or_path, {"type": "null"}]


def test_bad_list_error_keeps_its_shape(fake_embedder):
    text = call_over_wire({"texts_a": ["sentinel", 1]}).content[0].text
    assert "1 validation error" in text
    assert "texts_a.1\n" in text
    assert "sentinel" not in text


def test_missing_file(fake_embedder, tmp_path):
    path = str(tmp_path / "missing.json")
    message = similarity(path)["error"]
    assert message.startswith("cannot read texts_a: ")
    assert path in message
    assert similarity(["a"], path)["error"].startswith("cannot read texts_b: ")
    assert fake_embedder == []


def test_invalid_json(fake_embedder, tmp_path):
    path = tmp_path / "texts.json"
    path.write_text('["sentinel", oops]')
    message = similarity(str(path))["error"]
    assert message.startswith("cannot read texts_a: ")
    assert "sentinel" not in message
    assert fake_embedder == []


@pytest.mark.parametrize(
    "content",
    ['{"sentinel": 1}', '"sentinel"', "5", "null", '["sentinel", 1]', '[["sentinel"]]'],
)
def test_json_not_array_of_strings(fake_embedder, tmp_path, content):
    path = tmp_path / "texts.json"
    path.write_text(content)
    assert similarity(["a"], str(path)) == {
        "error": f"texts_b must be a JSON array of strings: {path}"
    }
    assert fake_embedder == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_permission_denied(fake_embedder, tmp_path):
    path = tmp_path / "texts.json"
    path.write_text('["a"]')
    path.chmod(0)
    assert "Permission denied" in similarity(str(path))["error"]
    assert fake_embedder == []


@pytest.mark.parametrize(
    "content",
    [b"\xff\xfe", b"[" * 200_000, b"[" + b"1" * 5000 + b"]"],
    ids=["not-utf8", "too-deep", "number-too-long"],
)
def test_unreadable_content(fake_embedder, tmp_path, content):
    path = tmp_path / "texts.json"
    path.write_bytes(content)
    assert similarity(str(path))["error"].startswith("cannot read texts_a: ")


@pytest.mark.parametrize("name", [".", "bad\0name.json"])
def test_unreadable_path(fake_embedder, tmp_path, name):
    assert similarity(f"{tmp_path}/{name}")["error"].startswith("cannot read texts_a: ")


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_path(fake_embedder, blank):
    assert similarity(blank) == {"error": "texts_a is blank"}
    assert similarity(["a"], blank) == {"error": "texts_b is blank"}
    assert fake_embedder == []


def test_file_texts_share_validation(fake_embedder, tmp_path, monkeypatch):
    assert similarity(write_texts(tmp_path / "a.json", [])) == {
        "error": "texts_a must not be empty"
    }
    assert similarity(["a"], write_texts(tmp_path / "b.json", ["a", " "])) == {
        "error": "texts_b[1] is blank"
    }
    monkeypatch.setenv("SIMILARITY_MAX_TEXTS", "2")
    assert similarity(["a"], write_texts(tmp_path / "c.json", ["a", "b"])) == {
        "error": "too many texts: 3 > 2"
    }
    assert fake_embedder == []


def test_limit_error_comes_before_reading(fake_embedder, tmp_path, monkeypatch):
    monkeypatch.setenv("SIMILARITY_MAX_TEXTS", "none")
    assert similarity(str(tmp_path / "missing.json")) == {
        "error": "SIMILARITY_MAX_TEXTS must be a positive integer"
    }


def test_csv_bad_file_writes_nothing(fake_embedder, tmp_path):
    out = tmp_path / "scores.csv"
    missing = str(tmp_path / "missing.json")
    message = similarity_csv(missing, str(out))["error"]
    assert message.startswith("cannot read texts_a: ")
    assert not out.exists()


def test_read_errors_are_data_over_wire(fake_embedder, tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("oops")

    async def scenario(session):
        for path in (str(tmp_path / "missing.json"), str(bad_json), "5"):
            result = await call_tool(session, {"texts_a": path})
            assert result.isError is False
            body = json.loads(result.content[0].text)
            assert body["error"].startswith("cannot read texts_a: ")
            await assert_serves(session)

    in_session(scenario)


def test_path_input_documented():
    section = readme_section("### `similarity-mcp`", "\n### ")
    for tool in asyncio.run(similarity_mcp.mcp.list_tools()):
        assert "path of a JSON file" in tool.description
    assert "path of a JSON file holding an array of strings" in section
    assert "`./null`" in section


def test_file_reads_in_security_section():
    section = readme_section("## Security", "\n## Layout")
    assert "reads only the JSON files a call names" in section


@pytest.mark.smoke
def test_smoke_path(tmp_path):
    path = write_texts(tmp_path / "texts.json", ["gato", "cat", "carro"])
    matrix = similarity(path)["matrix"]
    assert matrix[0][1] > matrix[0][2]
