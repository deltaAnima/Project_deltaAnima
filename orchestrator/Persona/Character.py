from typing import Any, Dict, Optional
from pathlib import Path
import time

from VisionLangHandler import VisionLangHandler
from Persona.PromptHandler import PromptHandler
from Persona.RAGHandler import Remembrance
from Persona.EmotionHandler import Enigmata
from Persona.FuliHandler import FuliHandler
import config


class Character:
    """
    A character's identity ties its prompt config, physics/RAG config, and
    the VL model it speaks through into one object. Everything besides
    chara_name and vl_handler auto-resolves from Config/<chara_name>/, so
    adding a new character is just adding a new folder there — this class
    never needs to change.
    """

    def __init__(self,
                 chara_name: str,
                 vl_handler: VisionLangHandler,
                 config_root: Optional[Path] = None,
                 session_timeout: float = 1800.0,
                 ):
        self.name: str = chara_name

        self.vl_handler: VisionLangHandler = vl_handler

        # single source of truth for every microservice address; see config.yaml
        self.server_ips: config.ServerIps = config.servers

        # session tracking: no explicit "session start" signal exists anywhere
        # upstream (Client/STT/edge), so we infer it ourselves — first request
        # from a given user_name, or one that arrives after `session_timeout`
        # seconds of silence from them, counts as a new session.
        self.session_timeout = session_timeout
        self._last_seen: Dict[str, float] = {}

        self.config_root: Path = config_root or (Path(__file__).parent / "Config")
        self.config_paths: Dict[str, Path] = self.__save_path__()

        # PromptHandler loads the character's prompt config and the shared VAD inference prompt config
        self.prompt_handler = PromptHandler(character=chara_name, config_root=self.config_root)

        # RAGHandler loads the character's RAG config and the shared RAG inference prompt config
        # 1. RAG config
        self.March_7th: Remembrance = Remembrance(character=chara_name, config_root=self.config_root)
        # 2. Emotion Engine  config
        self.Evernight: Enigmata = Enigmata(character=chara_name, config_root=self.config_root)
        # 3. DB server Handler
        self.Emanator: FuliHandler = FuliHandler(character=chara_name, 
                                                    config_root=self.config_root,
                                                    RAG_Handler=self.March_7th,
                                                    Emotion_Handler=self.Evernight)

    def get_character_prompt(self, user_name:str,
                                   user_input:str,
                                   context:Optional[str]=None) -> str:
        """
        Returns the character's prompt with the user's input and context filled in.
        Fetches memories + mood from Fuli in one round trip, then fills them
        into the Discord-text prompt template.
        """
        fuli_result = self.__call_Fuli__(user_input, user_name, context)
        return self.prompt_handler.get_reminh_prompt(
            user_name=user_name,
            memories=fuli_result["memories"],
            mood=fuli_result["emotion"] or ["calm"],
        )

    def __call_Fuli__(self, user_input:str,
                          user_name:str,
                          context:Optional[str]=None) -> Dict[str, Any]:
        """
        Single round trip to Fuli: sends the raw user_name/user_input/context
        plus this character's RAG/emotion policy as top-level fields — Fuli
        handles embedding, RAG search, and emotion computation itself.
        Returns {"memories": str, "emotion": ...}.
        """
        return self.Emanator.query_character_context(
            user_name=user_name,
            user_input=user_input,
            context=context,
            new_session=self.__is_new_session__(user_name),
        )

    def save_turn(self, user_name: str, persona_response: str) -> None:
        """
        Tells Fuli what this character actually said, closing out the
        memory buffer __call_Fuli__ opened for `user_name`. Must be
        called exactly once per turn (after the VL model produces its
        response) — Fuli refuses the next context request for the same
        user_name until this runs.
        """
        self.Emanator.save_context_memory(user_name=user_name, persona_response=persona_response)

    def __is_new_session__(self, session_key: str) -> bool:
        """
        True if this is the first request from `session_key` (usually
        user_name), or one arriving after session_timeout seconds of
        silence from them. Updates the last-seen timestamp as a side
        effect either way, so the timeout keeps sliding with activity.
        """
        now = time.monotonic()
        last_seen = self._last_seen.get(session_key)
        self._last_seen[session_key] = now
        return last_seen is None or (now - last_seen) > self.session_timeout

    def __save_path__(self) -> Dict[str, Path]:
        chara_dir = self.config_root / self.name
        return {
            "prompt_config_path": chara_dir / "prompt.yaml",
            "physics_weights_path": chara_dir / "physics_weights.yaml",
            "RAG_config_path": chara_dir / "RAG_config.yaml",
        }

    def __reload__(self, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Reloads every Config-driven handler this character owns.

        Pass a dict shaped like {"prompt": {...}, "physics": {...}, "rag": {...}}
        to hot-swap any of those handlers' config in memory — each sub-dict
        is validated against that handler's own current shape before it's
        applied, so a typo'd or malformed payload raises instead of
        silently corrupting the handler. Leave a key out (or omit `data`
        entirely) to just re-read that handler's yaml file from disk.
        """
        data = data or {}
        self.prompt_handler.reload(data.get("prompt"))
        self.Evernight.reload(data.get("physics"))
        self.March_7th.reload(data.get("rag"))
        print(f"--- Character '{self.name}' fully reloaded ---")
