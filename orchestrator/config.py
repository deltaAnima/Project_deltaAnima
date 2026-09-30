"""
Central config loader.

Pulls every environment-specific value (IPs, ports, model paths) out of
code and into config.yaml, so moving to a different machine or GPU box
never requires editing a .py file. Point ORCHESTRATOR_CONFIG at a
different path to use an alternate config (e.g. for a second rig).
"""

from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path
import asyncio
import os
import yaml


@dataclass
class ServerConfig:
    host: str
    port: int


@dataclass
class EdgeConfig:
    url: str
    role: str


@dataclass
class VLModelConfig:
    address: str
    port: str
    alias: str
    model_full: str
    temperature: float
    max_token: int
    top_p: float


@dataclass
class ServerIps:
    """host:port for every microservice on the LAN. Blank = not deployed yet."""

    Edge_server_ip: str
    Fuli_server_ip: str
    NV_5090_server_ip: str
    Embedding_server_ip: str
    Reranking_server_ip: str
    STT_server_ip: str
    TTS_server_ip: str
    Orchestrator_server_ip: str


@dataclass
class EmotionInferenceConfig:
    backend: str  # "openjev" | "5090"
    fallback_to_5090: bool


def _load_yaml() -> dict:
    config_path = Path(os.environ.get("ORCHESTRATOR_CONFIG", Path(__file__).parent / "config.yaml"))
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_raw = _load_yaml()

server = ServerConfig(**_raw["server"])

edge = EdgeConfig(
    # env var wins, so a deployment can repoint at a different edge hub
    # without touching config.yaml
    url=os.environ.get("EDGE_SERVER_URL", _raw["edge"]["url"]),
    role=_raw["edge"]["role"],
)

vl_model = VLModelConfig(**_raw["vl_model"])

emotion_inference = EmotionInferenceConfig(**_raw["emotion_inference"])

servers = ServerIps(**_raw["servers"])


async def _is_reachable(endpoint: str, timeout: float = 3.0) -> bool:
    """Plain TCP-connect liveness check against a 'host:port' endpoint."""
    if not endpoint or ":" not in endpoint:
        return False
    host, _, port = endpoint.partition(":")
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, int(port)), timeout=timeout)
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


async def test_servers() -> dict[str, bool]:
    """
    Liveness sweep over every entry in `servers`. This is a raw TCP connect,
    not a protocol-level health check, so it only proves "something is
    listening" — use a handler's own check_server()-style method when one
    exists (e.g. VisionLangHandler) for a more precise answer.
    """
    names = [f.name for f in fields(servers)]
    results = await asyncio.gather(*(_is_reachable(getattr(servers, name)) for name in names))
    return dict(zip(names, results))
