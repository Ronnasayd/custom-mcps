# custom-mcps

[MCP](https://modelcontextprotocol.io) servers (FastMCP) for **read-only** database queries, plus a Cloudflare Clef decision server and a local text similarity server. Each server runs over stdio; the database servers expose a single tool each.

| Command          | Tool                | Database   | Environment variables                                             |
| ---------------- | ------------------- | ---------- | ----------------------------------------------------------------- |
| `mongodb-mcp`    | `my_mcp_mongodb`    | MongoDB    | `MONGODB_URL`, `MONGODB_DATABASE`                                 |
| `mysql-mcp`      | `my_mcp_mysql`      | MySQL      | `MYSQL_URL` (URL with host, port, user, password, database)       |
| `postgresql-mcp` | `my_mcp_postgresql` | PostgreSQL | `POSTGRES_URL` (libpq DSN/URL)                                    |
| `sqlite-mcp`     | `my_mcp_sqlite`     | SQLite     | `SQLITEDB_PATH` (default: `database.db`)                          |
| `clef-mcp`       | `clef_*` (5 tools)  | -          | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `CLEF_MODEL`     |
| `similarity-mcp` | `similarity_matrix` | -          | `EMBEDDING_MODEL`, `SIMILARITY_MAX_TEXTS`, `FASTEMBED_CACHE_PATH` |

## Requirements

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
uv sync
```

## Usage

Run directly (the variable must already be exported in the environment):

```bash
uv run mysql-mcp
```

Or register it in your MCP client (e.g. `.mcp.json`):

```json
{
  "mcpServers": {
    "postgresql": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/custom-mcps", "postgresql-mcp"],
      "env": { "POSTGRES_URL": "<database-url>" }
    }
  }
}
```

## Tools

### `my_mcp_postgresql(query)` / `my_mcp_mysql(query)` / `my_mcp_sqlite(query)`

Run a SQL query and return `{"result": [...]}` or `{"error": "..."}`.

### `my_mcp_mongodb(collection, operation="find", query={}, projection={}, limit=100)`

`operation` accepts `find`, `find_one`, `count_documents` and `aggregate` (in which case `query` is the pipeline).
Values that are not JSON-serializable (e.g. `ObjectId`) are converted to `str`.

### `clef-mcp` (Cloudflare Clef decisions)

Wraps the hosted Clef models on Cloudflare Workers AI. Every tool returns the result or `{"error": "..."}`; the API token is never included in output.

- `clef_decide(state, questions, model=None)`: several typed questions in one call; returns the API `result` unchanged.
- `clef_choice(state, instructions, options, model=None)`: pick one option.
- `clef_score(state, instructions, levels, model=None)`: rate on an ordered scale.
- `clef_noul(state, instructions, model=None)`: probability that a statement holds.
- `clef_list_models()`: returns `{"models": ["clef", "clef-flash"], "default": ...}` without any network call.

`model` is `clef` or `clef-flash`. When omitted, `CLEF_MODEL` is used, then `clef`. `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` are required for the four decision tools.

### `similarity-mcp` (local text similarity)

`similarity_matrix(texts_a, texts_b=None)` scores pairs of texts by cosine similarity, from -1.0 to 1.0, rounded to 4 decimal places. It runs on the local machine and needs no credentials.

- One list: omit `texts_b` to compare `texts_a` with itself and get an N×N matrix.
- Two lists: pass `texts_b` to get an N×M matrix, with rows from `texts_a` and columns from `texts_b`.

The response is compact JSON with the fields `matrix` and `model`, such as `{"matrix":[[1.0,0.9909],[0.9909,1.0]],"model":"<model name>"}`. A cell's position is the index of its texts in the input lists. A bad call returns `{"error": "..."}` and the server keeps serving.

A call takes at most 300 texts across both lists. Set `SIMILARITY_MAX_TEXTS` to change the limit; it is read on each call.

`EMBEDDING_MODEL` selects the model, by default `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. The default model reads only the first 128 tokens of a text, so long texts that share an opening can score high.

The first start for a model downloads its files: the default model takes 241 MB on disk. They are kept in `FASTEMBED_CACHE_PATH`, by default `~/.cache/fastembed`. Later starts need no network.

The model runtime is an optional extra, so start the server with `--extra embeddings`:

```bash
uv run --directory /path/to/custom-mcps --extra embeddings similarity-mcp
```

The model loads before the server answers MCP traffic. When the load fails, the process exits with a non-zero status and a message on stderr: it names `--extra embeddings` when `fastembed` cannot be imported, and the cause otherwise.

A client can give up while the first download runs, so fetch the model from a terminal first. Run the command above once with the same `EMBEDDING_MODEL` and `FASTEMBED_CACHE_PATH` as the client, wait for `similarity-mcp: model ready` on stderr, stop it with Ctrl+D (Ctrl+C alone leaves it waiting on stdin), then register the server as in [Usage](#usage), with `"args": ["run", "--directory", "/path/to/custom-mcps", "--extra", "embeddings", "similarity-mcp"]`.

## Security (read-only databases)

The database servers are read-only; `clef-mcp` talks only to the Cloudflare API and touches no database. `similarity-mcp` touches no database and contacts the network only to download the model.

Protection is enforced on the database session, not only on the query:

- **PostgreSQL:** `set_session(readonly=True)`.
- **MySQL:** `SET SESSION TRANSACTION READ ONLY` + rollback at the end.
- **SQLite:** opened with `?mode=ro`; also rejects queries whose first token is `insert/update/delete/drop/alter/create/replace/truncate`.
- **MongoDB:** only the read operations listed above are accepted. The server does not enforce permissions itself — use a read-only database user.

In all cases, use credentials of a least-privilege user.

## Layout

```
src/databases/
  mongodb_mcp.py
  mysql_mcp.py
  postgresql_mcp.py
  sqlite_mcp.py
src/decision/
  clef_mcp.py
src/embeddings/
  similarity_mcp.py
```
