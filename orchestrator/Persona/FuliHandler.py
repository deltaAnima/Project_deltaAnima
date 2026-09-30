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
import requests

import config
from Persona.RAGHandler import Remembrance
from Persona.EmotionHandler import Enigmata
from Persona.RAG_schemas import CharacterContextResponse, ContextSaveResponse


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
        OCEAN/physics weights + backend flags (use_openjev/use_5090/fallback_to_5090).
        Fuli currently accepts this as unparsed JSON and ignores it —
        deltaEGO loads its own OCEAN/weights from its own yaml at startup
        — but we still send it so nothing needs to change once that's wired up.
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
        user_name: str,
        user_input: str,
        context: Optional[str] = None,
        new_session: bool = False,
        rag_policy: Optional[Dict[str, Any]] = None,
        emotion_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        One round trip to Fuli. Body matches schemas::FuliContextRequest
        exactly: user_name/user_input/context/new_session/rag_policy are
        top-level fields, emotion_policy_raw is untyped JSON. Fuli embeds
        the input, searches its DB, and runs the emotion engine itself,
        then reports back memories + mood as a CharacterContextResponse
        (Persona/RAG_schemas.py) — validated here, so a malformed or
        outdated Fuli response raises instead of silently corrupting the
        prompt downstream.

        `new_session` is Character's own guess (first message from this
        user, or one after a long enough gap) — MemoryRetriever owns the
        actual session id/lifetime on Fuli's side; this just forces it to
        mint a fresh one instead of reusing whatever's on file.

        Returns {"memories": str, "emotion": [emotion_term], "emotion_state":
        EmotionResult} — `emotion` is the short mood-tag list prompts want;
        `emotion_state` is deltaEGO's full VAD+resistance read-out, kept
        around for whatever wants the numbers later.
        """
        payload = {
            "user_name": user_name,
            "user_input": user_input,
            "context": context,
            "new_session": new_session,
            "config": {
                "rag_policy": rag_policy or self._default_rag_policy(),
                "emotion_policy_raw": emotion_policy or self._default_emotion_policy_raw(),
            },
        }
        raw_response = self._post("/character/context", payload)
        result = CharacterContextResponse.model_validate(raw_response)

        return {
            "memories": self.rag_handler.flatten_hits(result.hits, user_name),
            "emotion": [result.emotion.emotion_term],
            "emotion_state": result.emotion,
        }

    def save_context_memory(self, user_name: str, persona_response: str) -> ContextSaveResponse:
        """
        Closes out the memory buffer query_character_context() opened for
        `user_name` (POST /character/context_memory, schemas::FuliContextSaveRequest).
        Must be called exactly once per turn — Fuli refuses the *next*
        query_character_context() for the same user_name until this runs
        ("memory buffer already exists... a previous turn's
        HandleContextSaveRequest never ran"), confirmed against the live server.

        Returns the full saved-memory record Fuli reports back (the hashed
        user_id, faiss_id, session_id, the emotion snapshot at save time, etc.)
        """
        payload = {
            "user_name": user_name,
            "persona_name": self.character,
            "persona_response": persona_response,
        }
        raw_response = self._post("/character/context_memory", payload)
        return ContextSaveResponse.model_validate(raw_response)
