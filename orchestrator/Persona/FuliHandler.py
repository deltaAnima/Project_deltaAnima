"""
FuliHandler: the network transport to the Fuli C++ server (DB + emotion
engine, HTTP/REST + JSON). Fuli owns embedding, RAG search, and emotion
computation itself — we only hand it raw user input plus this character's
RAG/emotion policy, nested under "config" (confirmed against the live
server: schemas::FuliContextRequest's from_json reads rag_policy/
emotion_policy_raw from inside a top-level "config" object, not from the
request root); Remembrance and Enigmata are just where that policy lives,
not something we run locally before sending.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import logging
import time
import requests

import config
from Persona.RAGHandler import Remembrance
from Persona.EmotionHandler import Enigmata
from Persona.RAG_schemas import CharacterContextResponse, ContextSaveResponse

logger = logging.getLogger(__name__)


class FuliHandler:
    def __init__(
        self,
        character: str,
        RAG_Handler: Remembrance,
        Emotion_Handler: Enigmata,
        config_root: Optional[Path] = None,
        timeout: float = 10.0,
    ) -> None:
        self.character = character
        self.config_root: Path = config_root or (Path(__file__).parent / "Config")
        self.rag_handler = RAG_Handler
        self.emotion_handler = Emotion_Handler
        self.timeout = timeout

        self.endpoint = config.servers.Fuli_server_ip

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.endpoint:
            raise RuntimeError("Fuli_server_ip is not set (config.yaml -> servers)")
        response = requests.post(f"http://{self.endpoint}{path}", json=payload, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"[-] FULI Status {response.status_code}: {response.text}")
            raise e
        return response.json()

    def _default_rag_policy(self) -> Dict[str, Any]:
        """This character's RAGQueryOrder (Persona/RAG_schemas.py), no dense_vector — Fuli embeds the input itself."""
        return self.rag_handler.build_order().model_dump(mode="json")

    def _default_emotion_policy_raw(self) -> Dict[str, Any]:
        """
        OCEAN/physics weights + backend flags (use_laya/use_5090/fallback_to_5090).
        Fuli feeds this to deltaEGO on every request, so it is rebuilt per
        call — an Enigmata.reload() takes effect on the very next turn.
        """
        return {
            "OCEAN": self.emotion_handler.ocean,
            "AnalysisWeights": self.emotion_handler.analysis_weights,
            "FormulaWeights": self.emotion_handler.formula_weights,
            "AdvancedConfig": self.emotion_handler.advanced_config,
            **self.emotion_handler.backend_flags(),
        }

    def query_character_context(
        self,
        user_key: str,
        scope_key: str,
        user_name: str,
        user_input: str,
        context: Optional[str] = None,
        rag_policy: Optional[Dict[str, Any]] = None,
        emotion_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        One round trip to Fuli. Body matches schemas::FuliContextRequest
        exactly: user_name/user_key/scope_key/user_input/context are
        top-level fields, rag_policy/emotion_policy sit under "config".

        `user_key` ("<platform>:<platform_user_id>") is the stable identity
        Fuli hashes/locks/searches by; `scope_key` is where the user is
        talking (e.g. "discord:<guild_id>:<channel_id>"), which Fuli uses
        together with rag_policy.memory_config.scope to bound retrieval and
        to resolve the session; `user_name` is just the display name.

        Fuli embeds the input, searches its DB, and runs the emotion engine
        itself, then reports back memories + mood as a CharacterContextResponse
        (Persona/RAG_schemas.py) — validated here, so a malformed or
        outdated Fuli response raises instead of silently corrupting the
        prompt downstream.

        Returns {"memories": str, "recent_turns": [RecentTurn], "emotion":
        [emotion_term], "emotion_state": EmotionResult} — `recent_turns` is
        the latest saved turns, oldest first (the short-term history the VL
        call replays); `emotion` is the short mood-tag list prompts want;
        `emotion_state` is deltaEGO's full VAD+resistance read-out, kept
        around for whatever wants the numbers later.
        """
        payload = {
            "user_name": user_name,
            "user_key": user_key,
            "scope_key": scope_key,
            "user_input": user_input,
            "context": context,
            "config": {
                "rag_policy": rag_policy or self._default_rag_policy(),
                "emotion_policy": emotion_policy or self._default_emotion_policy_raw(),
            },
        }
        started = time.perf_counter()
        raw_response = self._post("/character/context", payload)
        result = CharacterContextResponse.model_validate(raw_response)
        if logger.isEnabledFor(logging.DEBUG):
            self._debug_context(user_key, scope_key, result, time.perf_counter() - started)

        return {
            "memories": self.rag_handler.flatten_hits(result.hits, user_name),
            "recent_turns": result.recent_turns,
            "emotion": [result.emotion.emotion_term],
            "emotion_state": result.emotion,
        }

    @staticmethod
    def _debug_context(user_key: str, scope_key: str, result: CharacterContextResponse, seconds: float) -> None:
        """DEBUG trace of one /character/context answer: what Fuli remembered and how it reads the mood."""
        emotion = result.emotion
        logger.debug(
            "Fuli context for %s (scope %s) in %.0f ms: %d memory hit(s), %d recent turn(s), emotion=%s (%s)  similarity=%s",
            user_key, scope_key, seconds * 1000, len(result.hits), len(result.recent_turns),
            emotion.emotion_term, emotion.current_state, emotion.similarity,
        )
        for hit in result.hits:
            logger.debug(
                "  memory #%s  score=%.3f\n  user : %s\n  reply: %s",
                hit.id, hit.score, hit.user_input[:300], hit.model_response[:300],
            )

    def save_context_memory(self, user_key: str, scope_key: str, user_name: str, persona_response: str) -> ContextSaveResponse:
        """
        Closes out the memory buffer query_character_context() opened for
        this user (POST /character/context_memory, schemas::FuliContextSaveRequest).
        Must be called exactly once per turn — Fuli refuses the *next*
        query_character_context() for the same user until this runs
        ("memory buffer already exists... a previous turn's
        HandleContextSaveRequest never ran"), confirmed against the live server.

        Returns the full saved-memory record Fuli reports back (the hashed
        user_id, faiss_id, session_id, the emotion snapshot at save time, etc.)
        """
        payload = {
            "user_name": user_name,
            "user_key": user_key,
            "scope_key": scope_key,
            "persona_name": self.character,
            "persona_response": persona_response,
        }
        raw_response = self._post("/character/context_memory", payload)
        return ContextSaveResponse.model_validate(raw_response)
