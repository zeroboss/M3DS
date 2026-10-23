# M3DS — Technical Roadmap

Durations assume 1–2 engineers and are indicative. Each phase ends with
something usable and an evaluation that proves it works across at least two
LLM providers.

## Phase 0 — Foundations (≈2 weeks) — ✅ built; live check pending API keys

- [x] Repo scaffolding: `uv`, ruff, pyright, pytest, pre-commit, CI (GitHub Actions with Postgres).
- [x] Docker Compose: Postgres + pgvector and the API. The background worker arrives with
      Phase 1, its first user; Langfuse (or any OTLP backend) is connected through
      `OTEL_EXPORTER_OTLP_ENDPOINT` rather than bundled.
- [x] Canonical message/request/response models; `LLMProvider` interface.
- [x] Adapters: **Anthropic** (Messages API, prompt caching, effort, server-side refusal
      fallback), **OpenAI** (stateless Responses API with encrypted reasoning replay),
      **Google Gemini** (`google-genai`, thought-signature replay) — generate, stream,
      structured output, tools.
- [x] Model capability registry (`config/models.yaml`) + config-driven routing and fallback.
- [x] Chat endpoint (REST + SSE) and CLI client with persistent session history.
- [x] Tracing of every LLM call (tokens, cost, latency) to the database and OpenTelemetry.

**Exit:** the same conversation runs on Claude, GPT and Gemini by changing one config
value — and can switch provider mid-conversation, including in the middle of a tool
exchange. Covered offline by the adapter mapping tests; the end-to-end check against the
real APIs is `tests/test_live.py` (`uv run pytest -m live`), still to be run with keys.

**Design notes carried into Phase 1**
- Provider-native content (thinking blocks, encrypted reasoning, thought signatures) is
  stored next to the portable parts and replayed only to the provider that produced it.
- History is append-only. Per-turn context (memories, retrieved knowledge) should be
  added after the stable, cacheable prefix instead of rewriting earlier turns, which keeps
  prompt caches warm and satisfies providers that validate replayed reasoning.

## Phase 1 — Memory (≈3 weeks)

- Storage schema for working, episodic and semantic memory (with temporal validity).
- Async write path: extraction → dedupe → ADD/UPDATE/SUPERSEDE/IGNORE.
- Read path with similarity + recency + importance scoring.
- Context Compiler v1 with token budgeting and cache-friendly ordering.
- Memory tools (`search_memory`, `remember`) for agentic models.
- Basic admin view: list / edit / delete memories.

**Exit:** facts told in session A are used correctly in session B, on a different provider.

## Phase 2 — Persona & linguistic style (≈2 weeks)

- Versioned StyleProfile (YAML import + API), exemplars, banned/preferred lexicon.
- Style signal capture (explicit feedback, thumbs, user rewrites).
- Reflection job proposing profile diffs; approve / auto-apply / rollback.
- Style-adherence LLM judge; per-provider calibration overrides.

**Exit:** a style eval scores ≥ target on all configured providers; style can be shifted through conversation alone.

## Phase 3 — Guidance → Directives (≈3 weeks)

- Correction / standing-instruction detector.
- Directive extraction (structured output) and user confirmation flow.
- Reconciliation (merge / specialise / supersede / conflict surfacing).
- Situational retrieval by trigger embedding + scope; injection with rationale and contrastive examples.
- Auto-generated regression test per directive, run across providers.

**Exit:** a correction given once while using one LLM is followed when the robot later runs on another LLM, verified by the regression suite.

## Phase 4 — Knowledge expansion (≈4 weeks)

- Ingestion workers: files (Docling/unstructured), URLs (trafilatura, Playwright fallback), crawl limits, re-crawl with change detection.
- Chunking with contextual headers; embedding; hybrid search + RRF + reranker; citations.
- Knowledge domains and distilled Notes (primers, glossaries).
- Autonomous research agent with budget, credibility scoring and a review step.
- Prompt-injection safeguards for ingested content.

**Exit:** given a topic and a few links, the robot builds a domain, answers domain questions with citations, and retrieval recall@k meets target.

## Phase 5 — Interfaces & integration (≈3 weeks)

- MCP server exposing memory, knowledge, persona and learning tools.
- Public REST + WebSocket/SSE API, Python client SDK.
- Admin console (Next.js): persona versions, directives review queue, memory browser, knowledge domains, traces, costs.
- Robot integration: voice (streaming STT/TTS) and/or ROS 2 bridge, latency tuning.

## Phase 6 — Hardening & scale (ongoing)

- Multi-tenant isolation (Postgres RLS), auth, encrypted secrets, audit log.
- Brain export/import; re-embedding migration tooling.
- Cost governance and per-robot quotas; caching strategy per provider.
- Continuous cross-provider eval dashboard; add new providers (local models via Ollama/vLLM).
- Optional: temporal knowledge graph (e.g. Graphiti-style) for entity-relationship reasoning.

## Key risks and mitigations

| Risk | Mitigation |
|---|---|
| Learning wrong rules or memories from ambiguous input | Confirmation step, confidence scores, review queue, versioning and rollback |
| Behaviour drift between providers | Exemplars over adjectives, per-provider calibration, cross-provider regression suite |
| Prompt bloat as memory/knowledge grows | Retrieval + token budgeting, consolidation, summarised episodes |
| Prompt injection via web/research content | Content treated as data only; directives/style writable only by authenticated users |
| Embedding-model lock-in | Store model metadata per vector; re-embedding job |
| Provider API churn | Thin adapters + contract tests per adapter |
