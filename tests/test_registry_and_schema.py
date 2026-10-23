from __future__ import annotations

from pathlib import Path

import pytest

from m3ds.llm.base import strict_json_schema
from m3ds.llm.registry import ModelRegistry, UnknownModelError
from m3ds.llm.types import Usage

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_registry_loads():
    reg = ModelRegistry.from_yaml(ROOT / "config" / "models.yaml")
    assert reg.route("chat")[0].provider == "anthropic"
    assert {m.provider for m in reg.route("chat")} == {"anthropic", "openai", "gemini"}


def test_resolve_alias_and_adhoc(registry: ModelRegistry):
    assert registry.resolve("primary").model_id == "p1-model"
    assert registry.resolve("p1:p1-model").alias == "primary"
    adhoc = registry.resolve("p9:new-model")
    assert (adhoc.provider, adhoc.model_id) == ("p9", "new-model")
    with pytest.raises(UnknownModelError):
        registry.resolve("nope")


def test_route_rejects_unknown_alias():
    with pytest.raises(ValueError, match="unknown models"):
        ModelRegistry.from_dict({"routes": {"chat": ["x"]}, "models": {}})


def test_cost_uses_cache_prices(registry: ModelRegistry):
    info = registry.resolve("primary").model_copy(deep=True)
    assert info.price is not None
    info.price.cache_read = 0.1
    cost = info.cost(Usage(input_tokens=1_000_000, cache_read_tokens=1_000_000, output_tokens=0))
    assert cost == pytest.approx(1.1)


def test_strict_json_schema_keeps_property_named_title():
    schema = {
        "title": "Doc",
        "type": "object",
        "properties": {"title": {"title": "Title", "type": "string"}, "n": {"type": "integer"}},
        "$defs": {
            "Sub": {"title": "Sub", "type": "object", "properties": {"x": {"type": "string"}}}
        },
    }
    out = strict_json_schema(schema)
    assert "title" not in out
    assert out["properties"]["title"] == {"type": "string"}
    assert out["required"] == ["title", "n"]
    assert out["additionalProperties"] is False
    assert out["$defs"]["Sub"]["additionalProperties"] is False
