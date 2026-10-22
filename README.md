# M3DS

An LLM-agnostic knowledge core for robots: persistent **memory**, a cultivable
**linguistic style**, **user-taught directives** that carry over between models,
and **specialised knowledge** learned from files, links and web research. Works
with Claude, GPT, Gemini and other models through online APIs.

- [Architecture & recommended stack](docs/ARCHITECTURE.md)
- [Technical roadmap](docs/ROADMAP.md)

**Stack in one line:** Python 3.12 + FastAPI + Pydantic, PostgreSQL with pgvector,
a thin in-house provider layer over the official Anthropic/OpenAI/Google SDKs,
background workers for learning, an MCP server for interoperability, and a
TypeScript admin console.
