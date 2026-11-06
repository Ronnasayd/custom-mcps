"""Tests for the similarity MCP server, with an injected embedder and a smoke tier."""

# why: pytest fixtures are injected by parameter name, and tests are docstring-exempt.
# pylint: disable=redefined-outer-name,missing-function-docstring,unused-argument
# pylint: disable=protected-access

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent


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
