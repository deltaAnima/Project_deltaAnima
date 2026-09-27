"""
Remembrance: a character's RAG query builder.

Loads the character's default RAG policy from Config/<character>/RAG_config.yaml
and turns raw user text + that policy into a RAGQueryOrder (Persona/RAG_schemas.py).
Remembrance never talks to the Fuli DB itself — FuliHandler owns that transport
and is the only thing that actually sends a built order anywhere.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

import config
from Persona.config_utils import load_yaml, validate_shape
from Persona.RAG_schemas import (
    RAGQueryOrder,
    MemoryQueryConfig,
    KnowledgeQueryConfig,
    DualTrackFusionConfig,
    RetrievedHit,
)


class Remembrance:
    def __init__(self, character: str, config_root: Optional[Path] = None) -> None:
        self.character = character
        self.config_root: Path = config_root or (Path(__file__).parent / "Config")
        self.config_path: Path = self.config_root / character / "RAG_config.yaml"

        self._load()

    def _load(self) -> None:
        raw = load_yaml(self.config_path)
        self.policy: Dict[str, Any] = raw.get("RAG_config", {})

        if not self.policy:
            raise ValueError(f"YAML format error!!!: {self.config_path}")

    def build_order(
        self,
        dense_vector: Optional[List[float]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        seed_entities: Optional[List[str]] = None,
        **overrides: Any,
    ) -> RAGQueryOrder:
        """Merge the character's yaml policy with per-call values into one RAGQueryOrder."""
        p = self.policy

        memory_cfg = MemoryQueryConfig(
            session_id=session_id,
            user_id=user_id,
            **p.get("memory_config", {}),
        )
        knowledge_cfg = KnowledgeQueryConfig(
            seed_entities=seed_entities,
            **p.get("knowledge_config", {}),
        )
        fusion_cfg = DualTrackFusionConfig(**p.get("fusion_config", {}))

        order_fields: Dict[str, Any] = {
            "target_domain": p.get("target_domain", "both"),
            "pipeline_level": p.get("pipeline_level", "v1_5_hybrid"),
            "dense_vector": dense_vector,
            "top_k": p.get("top_k", 5),
            "min_score_threshold": p.get("min_score_threshold", 0.6),
            "memory_config": memory_cfg,
            "knowledge_config": knowledge_cfg,
            "fusion_config": fusion_cfg,
            "granularity": p.get("granularity", "raw_chunk"),
            "context_window_size": p.get("context_window_size", 1),
            "enable_rerank": p.get("enable_rerank", False),
        }
        order_fields.update(overrides)
        return RAGQueryOrder(**order_fields)

    def embed_query(self, text: str) -> List[float]:
        """
        Fetches a dense embedding for `text`.

        Assumes an OpenAI-compatible /v1/embeddings endpoint, mirroring the
        convention VisionLangHandler already uses for the VL model server.
        Adjust the endpoint/response parsing here if the embedding server's
        actual API differs.
        """
        endpoint = config.servers.Embedding_server_ip
        if not endpoint:
            raise RuntimeError("Embedding_server_ip is not set (config.yaml -> servers)")

        response = requests.post(
            f"http://{endpoint}/v1/embeddings",
            json={"input": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def flatten_hits(self, hits: List[RetrievedHit], user_name: str) -> str:
        """Flattens Fuli's retrieved past turns into "<name>: <text>" lines for prompt injection."""
        lines = []
        for hit in hits:
            if hit.user_input:
                lines.append(f"{user_name}: {hit.user_input}\n{self.character}: {hit.model_response}")
            else:
                lines.append(f"{self.character}: {hit.model_response}")
        return "\n".join(lines)

    def reload(self, data: Optional[Dict[str, Any]] = None) -> None:
        """
        With no `data`, re-reads RAG_config.yaml from disk. With `data`,
        hot-swaps it in memory instead — `data` must be shaped like
        {"RAG_config": {...}}, matching this handler's own currently-loaded
        shape exactly.
        """
        if data is None:
            self._load()
        else:
            reference = {"RAG_config": self.policy}
            validate_shape(data, reference)
            self.policy = data["RAG_config"]
        print(f"--- RAG Config Reloaded ({self.character}) ---")
