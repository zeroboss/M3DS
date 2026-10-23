"""Command-line interface: serve the API, chat in the terminal, inspect models."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

import typer

from m3ds.config import get_settings

app = typer.Typer(help="M3DS knowledge core", no_args_is_help=True)
models_app = typer.Typer(help="Inspect the model registry", no_args_is_help=True)
app.add_typer(models_app, name="models")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("m3ds.api.app:create_app", factory=True, host=host, port=port, reload=reload)


@app.command()
def chat(
    model: Annotated[str | None, typer.Option(help="Model alias or provider:model-id")] = None,
    conversation: Annotated[str | None, typer.Option(help="Resume a conversation id")] = None,
) -> None:
    """Interactive chat with persistent history. Type /model <alias> to switch models."""
    asyncio.run(_chat(model, conversation))


async def _chat(model: str | None, conversation_id: str | None) -> None:
    from m3ds.chat.service import ChatService
    from m3ds.llm.base import ProviderError
    from m3ds.llm.factory import build_gateway
    from m3ds.llm.types import StreamDone, TextDelta
    from m3ds.storage.db import DatabaseRecorder, create_schema, make_engine, make_sessionmaker

    settings = get_settings()
    engine = make_engine(settings.database_url)
    if settings.auto_create_schema:
        await create_schema(engine)
    sessionmaker = make_sessionmaker(engine)
    gateway = build_gateway(settings, [DatabaseRecorder(sessionmaker)])
    svc = ChatService(gateway, sessionmaker, settings.system_prompt, settings.history_max_messages)
    if conversation_id is None:
        conversation_id = (await svc.create_conversation()).id
    typer.secho(f"conversation {conversation_id}  (/model <alias>, /quit)", dim=True)
    try:
        while True:
            try:
                text = (await asyncio.to_thread(input, "you> ")).strip()
            except EOFError:
                break
            if not text:
                continue
            if text in ("/quit", "/exit"):
                break
            if text.startswith("/model"):
                model = text.split(maxsplit=1)[1] if " " in text else None
                typer.secho(f"model = {model or '(route default)'}", dim=True)
                continue
            typer.echo("bot> ", nl=False)
            try:
                async for event in svc.send_stream(conversation_id, text, model):
                    if isinstance(event, TextDelta):
                        typer.echo(event.text, nl=False)
                    elif isinstance(event, StreamDone):
                        r, u = event.response, event.response.usage
                        cost = f" ${r.cost_usd:.5f}" if r.cost_usd is not None else ""
                        typer.secho(
                            f"\n  [{r.alias} {r.model} in={u.input_tokens}"
                            f" cached={u.cache_read_tokens} out={u.output_tokens}{cost}]",
                            dim=True,
                        )
            except ProviderError as exc:
                typer.secho(f"\nerror: {exc}", fg="red")
    finally:
        await gateway.aclose()
        await engine.dispose()


@app.command()
def ask(
    prompt: str,
    model: Annotated[list[str] | None, typer.Option("--model", "-m", help="Repeatable")] = None,
    all_models: Annotated[
        bool, typer.Option("--all", help="Ask every available chat model")
    ] = False,
) -> None:
    """One-shot question (no history). Use --all to compare providers side by side."""
    asyncio.run(_ask(prompt, model or [], all_models))


async def _ask(prompt: str, models: list[str], all_models: bool) -> None:
    from m3ds.llm.base import ProviderError
    from m3ds.llm.factory import build_gateway
    from m3ds.llm.types import LLMRequest, Message, SystemBlock

    settings = get_settings()
    gateway = build_gateway(settings)
    if all_models:
        models = [m.alias for m in gateway.registry.route("chat") if gateway.is_available(m)]
    targets: list[str | None] = list(models) or [None]
    req = LLMRequest(
        messages=[Message.user(prompt)], system=[SystemBlock(text=settings.system_prompt)]
    )
    try:
        results = await asyncio.gather(
            *(gateway.generate(req.model_copy(update={"model": m})) for m in targets),
            return_exceptions=True,
        )
        for target, result in zip(targets, results, strict=True):
            typer.secho(f"── {target or 'default'}", bold=True)
            if isinstance(result, ProviderError):
                typer.secho(f"error: {result}", fg="red")
            elif isinstance(result, BaseException):
                raise result
            else:
                typer.echo(result.text)
                out = result.usage.output_tokens
                typer.secho(f"  [{result.model} {result.latency_ms:.0f}ms out={out}]", dim=True)
    finally:
        await gateway.aclose()


@models_app.command("list")
def models_list() -> None:
    """Show registry aliases and whether their provider is configured."""
    from m3ds.llm.factory import build_gateway

    gateway = build_gateway(get_settings())
    for m in gateway.registry.models.values():
        mark = "✓" if gateway.is_available(m) else "·"
        typer.echo(f"{mark} {m.alias:<14} {m.provider:<10} {m.model_id}")
    for task, chain in gateway.registry.routes.items():
        typer.echo(f"route {task}: {' → '.join(chain)}")


@models_app.command("probe")
def models_probe() -> None:
    """Ask each configured provider which model ids it serves and flag registry
    entries that are not among them."""
    asyncio.run(_probe())


async def _probe() -> None:
    from m3ds.llm.base import ProviderError
    from m3ds.llm.factory import build_gateway

    gateway = build_gateway(get_settings())
    try:
        for name, provider in gateway.providers.items():
            if name == "fake":
                continue
            try:
                served = {s.removeprefix("models/") for s in await provider.list_models()}
            except ProviderError as exc:
                typer.secho(f"{name}: {exc}", fg="red")
                continue
            typer.secho(f"{name}: {len(served)} models", bold=True)
            for m in gateway.registry.models.values():
                if m.provider == name:
                    ok = m.model_id in served
                    typer.secho(
                        f"  {'✓' if ok else '✗'} {m.alias:<14} {m.model_id}",
                        fg=None if ok else "yellow",
                    )
    finally:
        await gateway.aclose()


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    app()


if __name__ == "__main__":
    main()
