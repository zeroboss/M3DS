# M3DS

An LLM-agnostic knowledge core for robots: persistent **memory**, a cultivable
**linguistic style**, **user-taught directives** that carry over between models,
and **specialised knowledge** learned from files, links and web research. Works
with Claude, GPT, Gemini and other models through online APIs.

- [Architecture & recommended stack](docs/ARCHITECTURE.md)
- [Technical roadmap](docs/ROADMAP.md)

**Stack in one line:** Python 3.12+ with FastAPI and Pydantic, PostgreSQL with pgvector,
a thin in-house provider layer over the official Anthropic/OpenAI/Google SDKs,
background workers for learning, an MCP server for interoperability, and a
TypeScript admin console.

## Status

**Phase 0 (foundations)** is in place: one canonical message format, adapters for
Anthropic (Messages API), OpenAI (Responses API) and Gemini (`google-genai`),
a model registry with routing and cross-provider fallback, validated structured
output, persistent conversations, call tracing (tokens, cost, latency) in the
database and over OpenTelemetry, a REST/SSE API, and a CLI.

## Quick start

```bash
uv sync                      # Python 3.12+, https://docs.astral.sh/uv/
cp .env.example .env         # add ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY

uv run m3ds models list      # which aliases are usable with your keys
uv run m3ds models probe     # check registry model ids against each provider's live list
uv run m3ds ask "Introduce yourself in one sentence" --all   # same prompt, every provider
uv run m3ds chat             # persistent chat; `/model gpt` switches model mid-conversation
uv run m3ds serve            # HTTP API on http://127.0.0.1:8000/docs
```

Without any API key you can try everything with the offline `fake` model
(`m3ds chat --model fake`).

With Postgres (recommended beyond local experiments):

```bash
docker compose up -d db
export M3DS_DATABASE_URL=postgresql+asyncpg://m3ds:m3ds@localhost:5432/m3ds M3DS_AUTO_CREATE_SCHEMA=false
uv run alembic upgrade head
# or run the whole stack: docker compose up --build
```

### HTTP API

| Method & path | Purpose |
|---|---|
| `GET /healthz` | Liveness |
| `GET /v1/models` | Registry aliases and availability |
| `POST /v1/conversations` | Create a conversation (`robot_id`, `user_id`, `title`) |
| `GET /v1/conversations/{id}/messages` | History |
| `POST /v1/conversations/{id}/messages` | Send `{"text", "model"?, "stream"?}`; `stream: true` returns Server-Sent Events (`delta`, `done`, `error`) |

Set `M3DS_API_KEY` to require an `X-API-Key` header.

### Models and routing

`config/models.yaml` maps aliases (`claude`, `gpt`, `gemini`, …) to provider model
ids, capabilities and prices, and defines routes per task (`chat`, `extract`,
`judge`). A request without an explicit model goes to the first configured model on
its route and falls back to the next one on rate limits, overload or network errors.
An explicitly requested model never silently switches. Model ids change often:
run `m3ds models probe` and edit the YAML rather than the code.

## Development

```bash
uv run pytest                # unit + API tests (offline, SQLite)
M3DS_TEST_DATABASE_URL=postgresql+asyncpg://... uv run pytest   # same tests on Postgres
uv run pytest -m live        # real API contract tests (needs keys; costs a little)
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

Layout: `src/m3ds/llm` (types, registry, gateway, providers), `src/m3ds/chat`
(conversation service), `src/m3ds/storage` (SQLAlchemy models), `src/m3ds/api`
(FastAPI), `src/m3ds/cli.py`, `migrations/` (Alembic).
