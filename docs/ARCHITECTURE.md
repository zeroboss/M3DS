# M3DS — LLM-Agnostic Robot Knowledge Core: Stack & Architecture

## 1. Design principle

**The LLM is a stateless reasoning engine. Everything that makes the robot *this*
robot lives in the knowledge core, outside any model.**

Claude, GPT and Gemini are interchangeable "CPUs". Persona, memories, learned
rules of behaviour and domain knowledge are stored as **plain data in your own
schema** (natural language + structured metadata + embeddings). Before each LLM
call the core *compiles* the relevant slice of that data into a prompt for
whichever provider is in use. Because nothing important depends on provider-side
features (fine-tunes, provider memory, provider threads), you can switch models
per request without the robot losing its identity.

```
            ┌──────────────────────────────────────────────────────────┐
 user /     │                     Knowledge Core                       │
 robot ───► │  Turn Orchestrator                                       │
 client     │    1. retrieve  ──► Persona/Style │ Directives │ Memory  │
            │                      Knowledge    │ Session history      │
            │    2. assemble  ──► Context Compiler (token budget,      │
            │                     cache-friendly ordering)             │
            │    3. call      ──► LLM Gateway ─► Claude │ GPT │ Gemini  │
            │    4. respond   ◄── (streaming)                          │
            │    5. learn (async) ─► extract memories, detect          │
            │                        corrections → directives,         │
            │                        style feedback                    │
            └──────────────────────────────────────────────────────────┘
                     │                         ▲
                     ▼                         │
            Background workers: ingestion, web research, memory
            consolidation, style reflection, evaluation runs
```

## 2. Recommended stack

| Layer | Choice | Why |
|---|---|---|
| Core language | **Python 3.12+** | Best ecosystem for embeddings, document parsing, scraping, evals; first-class official SDKs from Anthropic, OpenAI and Google. |
| Packaging / tooling | `uv`, `ruff`, `pyright` (or mypy), `pytest` | Fast, modern, reproducible. |
| API service | **FastAPI** + Pydantic v2 (async) | Typed schemas shared across API, storage and LLM structured outputs; WebSocket/SSE for streaming. |
| LLM access | **Own thin `LLMProvider` interface** with adapters over the official SDKs (`anthropic`, `openai`, `google-genai`); optional LiteLLM adapter for the long tail (Mistral, DeepSeek, Qwen, Ollama/local) | You keep control of the canonical message format, tool-calling and caching semantics; no framework lock-in. |
| Primary database | **PostgreSQL 16+ with `pgvector`** | One store for relational data, vectors (HNSW), full-text search (`tsvector`, or ParadeDB `pg_search` for BM25) and JSONB. Transactions across memory + metadata. Easy backups. |
| ORM / migrations | SQLAlchemy 2.0 (async) + Alembic | Mature, explicit. |
| Background jobs | Procrastinate (Postgres-backed, no extra infra) — or Arq/Celery + Redis if you already run Redis | Ingestion, research, consolidation, evals run off the hot path. |
| Cache / rate limits | Redis (optional at first) | Session state, per-provider rate limiting. |
| Embeddings | **One fixed embedding model, independent of the chat LLM**: `bge-m3` self-hosted (multilingual, dense+sparse) or a hosted API (OpenAI `text-embedding-3-large`, Voyage, Gemini embeddings) | Embeddings must be stable across chat-model switches. Store the embedding model name/version on every vector so you can re-embed later. |
| Reranker | `bge-reranker-v2-m3` (local) or Cohere/Voyage rerank API | Big quality jump for knowledge retrieval. |
| Ingestion | `httpx`, `trafilatura` (web pages), Playwright (JS-heavy pages, already common in CI images), **Docling** or `unstructured` (PDF/DOCX/PPTX), `pymupdf` | Robust text + structure extraction. |
| Web search | Tavily / Brave Search API / Exa behind a `SearchProvider` interface | Provider-native search tools (Claude/OpenAI/Gemini web search) can be optional extras, but keep a model-independent path. |
| Tool / agent interop | **MCP server** (official `mcp` Python SDK) + REST API | Any MCP-capable client or agent (Claude, ChatGPT, Gemini-based agents, IDEs) can read/write the robot's memory and knowledge directly. |
| Observability | OpenTelemetry + **Langfuse** (self-hostable) | Trace every assembled context: "why did the robot say that?" must be answerable. |
| Admin UI | TypeScript + React/Next.js (start with Streamlit/Gradio for the prototype) | Humans must be able to inspect, edit, approve and roll back memories, directives and style versions. |
| Deployment | Docker Compose → Kubernetes later | Postgres + API + worker is enough for a long time. |
| Robot side (if physical) | ROS 2 bridge node (`rclpy`), streaming STT/TTS | Core stays a service; the robot is just another client. |

**Why not TypeScript for the core?** It is viable (the Vercel AI SDK has a good
multi-provider abstraction), but document parsing, local embedding/reranking
models and evaluation tooling are much stronger in Python. Use TypeScript for
the UI only.

**Why not LangChain/LlamaIndex as the core?** Your differentiating features
(directives, style cultivation, memory consolidation) are custom logic; a heavy
framework adds abstraction churn without solving them. Borrow individual
utilities if useful. Projects worth studying for ideas: Mem0 (memory
add/update/delete decisions), Letta/MemGPT (self-editing memory), Zep/Graphiti
(temporal knowledge graphs), LangMem.

## 3. Core modules

### 3.1 LLM Gateway (cross-provider layer)

```python
class LLMProvider(ABC):                     # src/m3ds/llm/base.py
    name: str
    async def generate(self, req: LLMRequest, model: ModelInfo) -> LLMResponse: ...
    def stream(self, req: LLMRequest, model: ModelInfo) -> AsyncIterator[StreamEvent]: ...
    async def list_models(self) -> list[str]: ...

class LLMGateway:                           # src/m3ds/llm/gateway.py
    async def generate(self, req, task="chat") -> LLMResponse: ...      # routing + fallback
    def stream(self, req, task="chat") -> AsyncIterator[StreamEvent]: ...
    async def structured(self, req, schema: type[T], task="extract") -> T: ...
```

- **Canonical message format** (your own Pydantic models: system blocks, user/assistant
  turns, tool calls/results, images). Adapters translate to Anthropic Messages,
  OpenAI Responses/Chat Completions and Gemini `generateContent`.
- **Capability registry** per model (context window, max output, tool use, structured
  output, prompt caching, vision, price). Model IDs live in config, never in code.
- **Routing policy**: pick a model per task (`chat`, `extract`, `summarize`, `research`,
  `judge`), with fallbacks on error/rate-limit. Cheap/fast models for background
  extraction, the user's chosen model for conversation.
- **Structured output** for all internal jobs: JSON schema → validate with Pydantic →
  retry with the validation error on failure. Works the same on every provider.
- **Prompt caching**: the Context Compiler emits stable content first (identity,
  style, global directives) so Anthropic/OpenAI/Gemini caching kicks in.
- **Budget control**: per-user/per-robot token and cost ceilings, recorded per call.

### 3.2 Persona & Linguistic Style

```
StyleProfile (versioned, rollback-able)
  identity:        name, role, backstory, values
  language:        primary language(s), dialect, code-switching rules
  register:        formality, warmth, humour, assertiveness (0–1 scales + prose)
  form:            sentence length, verbosity, formatting, emoji policy
  lexicon:         preferred phrases, catch-phrases, banned words/phrases
  exemplars[]:     {situation, ideal_response}   ← the most portable part
  provider_overrides{provider/model: extra instructions}  ← calibration
```

- **Pre-defined**: author the profile (YAML/JSON or admin UI).
- **Cultivated through conversation**: every turn can emit *style signals* — explicit
  ("be less formal", "stop using lists") and implicit (user rewrites a reply,
  thumbs-down). A periodic *style reflection* job aggregates signals, proposes a
  diff to the profile (and new exemplars taken from replies the user liked), and
  either auto-applies low-risk changes or asks for approval. Every change is a new
  version.
- **Cross-LLM consistency**: adjectives are interpreted differently by different
  models ("concise" for one model is verbose for another), so exemplars carry
  most of the weight, and `provider_overrides` stores per-model calibration learned
  from evals (§3.6).

### 3.3 Memory

| Type | Content | Example |
|---|---|---|
| Working | Current session turns + rolling summary | last 10 turns |
| Episodic | Summaries of past conversations/events, timestamped | "2026-10-20: helped user plan Kyoto trip" |
| Semantic / profile | Durable facts about the user, the world, the robot itself | "User is vegetarian" |
| Procedural | → see Directives (§3.4) | |

Memory record: `id, owner(robot_id, user_id), type, content, embedding,
importance, confidence, source_turn_id, created_at, last_accessed_at,
valid_from, valid_to, superseded_by`.

- **Write path (async, after each turn)**: extract candidate memories with a cheap
  model → find near-duplicates by vector similarity → LLM decides
  `ADD / UPDATE / SUPERSEDE / IGNORE` (Mem0-style) → store with provenance.
  Contradictions are handled by *superseding* (set `valid_to`), not deleting, so
  history stays auditable.
- **Read path**: score = `α·similarity + β·recency_decay + γ·importance`
  (Generative-Agents style), plus always-included core profile facts.
- **Consolidation job**: merge duplicates, summarise old episodes, decay
  unimportant items, flag conflicts for review.
- **Two retrieval modes**, both supported:
  - *push*: the orchestrator pre-fetches memories into the prompt (works on every model);
  - *pull*: memory exposed as tools (`search_memory`, `remember`) for agentic models.

### 3.4 Guidance → Directives (user-steered reasoning that persists)

This is the feature that makes "teach it once, it behaves that way on every
LLM" work. A user correction is turned into an explicit, model-independent
**Directive**:

```
Directive
  trigger:      natural-language description of the situation   (+ embedding)
  instruction:  what to do / how to reason
  rationale:    why (helps the model generalise correctly)
  examples[]:   {context, rejected_response, preferred_response}
  scope:        global | domain:<name> | user:<id>
  priority, status (proposed | active | retired), version, supersedes
  provenance:   turn ids, author, created_at
```

Lifecycle:
1. **Detect** – a lightweight classifier on each user turn: is this feedback,
   a correction, or a standing instruction ("from now on…", "no, first check…")?
2. **Extract** – structured-output call turns the exchange into a Directive draft.
3. **Confirm** – the robot echoes it back ("From now on, when you ask about X, I'll
   first Y — keep this?"). Explicit confirmation avoids learning wrong rules.
4. **Reconcile** – compare with existing directives (similarity + LLM check) to merge,
   specialise or supersede; surface true conflicts to the user.
5. **Apply** – at inference, retrieve directives whose *trigger* matches the
   current situation (vector search + scope filter); global ones are always
   included. They're injected as numbered rules with their rationale, plus one
   contrastive example when budget allows.
6. **Verify** – every directive becomes a **regression test** (its example context
   → assert behaviour with an LLM judge), run against every configured provider.
   This is how you *prove* the same reasoning carries across Claude, GPT and Gemini.

### 3.5 Specialised knowledge expansion

Data model: `KnowledgeDomain → Source (file | url | search) → Document → Chunk`,
plus **distilled Notes** (LLM-written topic primers, glossaries, FAQs) linked
back to the chunks they cite.

Ingestion pipeline (worker):
`fetch → extract (trafilatura / Docling) → clean → structure-aware chunking →
contextual header per chunk (doc title + section summary) → embed → index
(vector + full-text) → store provenance (url, retrieved_at, hash, trust level)`.

Three ways to learn:
1. **User resources** – uploaded files (PDF, DOCX, Markdown, slides, transcripts).
2. **User links** – single page or depth/domain-limited crawl; respect robots.txt;
   scheduled re-crawl with content-hash change detection.
3. **Autonomous research** – given a topic and a budget: plan sub-questions → search
   API → fetch top results → score relevance & source credibility → ingest →
   gap analysis → iterate → write a domain primer + glossary with citations →
   user reviews/approves the domain.

Retrieval: hybrid (vector + BM25/full-text) → reciprocal-rank fusion → rerank →
top-k with citations. Notes are retrieved alongside raw chunks for better
"understanding"-level answers.

**Security**: ingested content is *untrusted data*. It is always placed in the
prompt inside clearly delimited reference blocks, never treated as instructions,
and it can never create or modify directives or style — only an authenticated
user can (prompt-injection defence).

### 3.6 Evaluation & observability

- Golden conversation sets per robot: style adherence, memory recall, directive
  compliance, grounded-answer accuracy — scored by an LLM judge + simple metrics.
- Run the same suite across providers/models on every change → a
  **cross-LLM consistency report**; feeds `provider_overrides` calibration.
- Retrieval metrics (recall@k, MRR) on a labelled query set.
- Full tracing of each turn: retrieved items, assembled prompt, model, tokens, cost, latency.

### 3.7 Context Compiler (turn assembly)

Order (stable → volatile, for cache hits) and budgeted by the target model's window:

1. Core identity + style profile + exemplars *(cacheable)*
2. Global directives *(cacheable)*
3. Situational directives retrieved for this turn
4. User profile memories
5. Retrieved knowledge (delimited, with citations, marked as reference data)
6. Relevant episodic memories
7. Session summary + recent turns
8. Current user message

## 4. Cross-cutting concerns

- **Multi-tenancy**: every row carries `robot_id` (and `user_id` where relevant);
  row-level security in Postgres.
- **Portability**: a versioned "brain export" (JSONL of persona versions, memories,
  directives, sources, notes + embedding-model metadata) so a robot's mind can be
  backed up, moved, or re-embedded with a different embedding model.
- **Privacy**: user can list/edit/delete memories; PII tagging; API keys encrypted
  at rest; optional per-memory "never send to provider X" flags.
- **Latency (voice/physical robots)**: streaming responses, retrieval under ~200 ms,
  learning always off the hot path.

## 5. Proposed repository layout

```
m3ds/
  core/            # Pydantic domain models: StyleProfile, Memory, Directive, Knowledge*
  llm/             # LLMProvider protocol, adapters (anthropic, openai, gemini, litellm), router, registry
  context/         # Context Compiler, token budgeting
  persona/         # style profile store, signal extraction, reflection job
  memory/          # write/read paths, scoring, consolidation
  directives/      # detection, extraction, reconciliation, retrieval
  knowledge/       # ingestion, chunking, hybrid retrieval, research agent, notes
  search/          # SearchProvider adapters
  storage/         # SQLAlchemy models, repositories, Alembic migrations
  workers/         # background job definitions
  api/             # FastAPI app (REST + WebSocket/SSE)
  mcp_server/      # MCP tools/resources over the same services
  evals/           # golden sets, judges, cross-provider runner
ui/                # admin console (Next.js)
deploy/            # docker-compose, k8s manifests
```

See [ROADMAP.md](ROADMAP.md) for the phased delivery plan.
