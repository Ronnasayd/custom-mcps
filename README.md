# custom-mcps

[MCP](https://modelcontextprotocol.io) servers (FastMCP) for **read-only** database queries, plus a Cloudflare Clef decision server. Each server runs over stdio; the database servers expose a single tool each.

| Command          | Tool                | Database   | Environment variables                                         |
| ---------------- | ------------------- | ---------- | ------------------------------------------------------------- |
| `mongodb-mcp`    | `my_mcp_mongodb`    | MongoDB    | `MONGODB_URL`, `MONGODB_DATABASE`                             |
| `mysql-mcp`      | `my_mcp_mysql`      | MySQL      | `MYSQL_URL` (URL with host, port, user, password, database)   |
| `postgresql-mcp` | `my_mcp_postgresql` | PostgreSQL | `POSTGRES_URL` (libpq DSN/URL)                                |
| `sqlite-mcp`     | `my_mcp_sqlite`     | SQLite     | `SQLITEDB_PATH` (default: `database.db`)                      |
| `clef-mcp`       | `clef_*` (5 tools)  | -          | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `CLEF_MODEL` |

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

## Security (read-only databases)

The database servers are read-only; `clef-mcp` talks only to the Cloudflare API and touches no database.

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
```
