from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from VisionLangHandler import VisionLangHandler
from Persona.PromptHandler import PromptHandler
from Persona.RAGHandler import Remembrance
from Persona.EmotionHandler import Enigmata
from Persona.FuliHandler import FuliHandler
from Persona.RAG_schemas import ContextSaveResponse, RecentTurn
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
                 ):
        self.name: str = chara_name

        self.vl_handler: VisionLangHandler = vl_handler

        # single source of truth for every microservice address; see config.yaml
        self.server_ips: config.ServerIps = config.servers

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

    def get_character_prompt(self, user_key:str,
                                   scope_key:str,
                                   user_name:str,
                                   user_input:str,
                                   context:Optional[str]=None,
                                   platform:Optional[str]=None) -> Tuple[str, List[Dict[str, str]]]:
        """
        Returns (system prompt, history): the character's prompt with the
        user's input and context filled in, plus Fuli's recent turns as
        OpenAI-style user/assistant messages for the VL call to replay
        before the current input. Fetches memories + mood + recent turns
        from Fuli in one round trip.

        `user_key` ("<platform>:<platform_user_id>") identifies the user,
        `scope_key` identifies where they're talking (e.g. a Discord channel);
        `user_name` is only the display name shown in the prompt. Sessions
        are resolved by Fuli from user_key + scope_key. `platform` picks the
        prompt template ("discord" → text-chat prompt).
        """
        fuli_result = self.__call_Fuli__(user_input, user_key, scope_key, user_name, context)
        # Discord gets the text-chat prompt; everything else (Unity/TTS) the spoken one
        build_prompt = (
            self.prompt_handler.get_discord_Text_prompt
            if platform == "discord"
            else self.prompt_handler.get_reminh_prompt
        )
        system_prompt = build_prompt(
            user_name=user_name,
            memories=fuli_result["memories"],
            mood=fuli_result["emotion"] or ["calm"],
        )
        return system_prompt, self.__history_messages__(fuli_result["recent_turns"], user_key)

    @staticmethod
    def __history_messages__(recent_turns: List[RecentTurn], user_key: str) -> List[Dict[str, str]]:
        """
        Fuli's recent turns (oldest first) as alternating user/assistant
        messages. Turns from someone other than the current user (a shared
        channel) get their name prefixed so the model can tell speakers apart.
        """
        messages: List[Dict[str, str]] = []
        for turn in recent_turns:
            user_text = turn.user_input if turn.user_key == user_key else f"{turn.user_name}: {turn.user_input}"
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": turn.model_response})
        return messages

    def __call_Fuli__(self, user_input:str,
                          user_key:str,
                          scope_key:str,
                          user_name:str,
                          context:Optional[str]=None) -> Dict[str, Any]:
        """
        Single round trip to Fuli: sends the raw user_key/scope_key/user_name/
        user_input/context plus this character's RAG/emotion policy — Fuli
        handles embedding, RAG search, and emotion computation itself.
        Returns {"memories": str, "recent_turns": [RecentTurn], "emotion": ...}.
        """
        return self.Emanator.query_character_context(
            user_key=user_key,
            scope_key=scope_key,
            user_name=user_name,
            user_input=user_input,
            context=context,
        )

    def save_turn(self, user_key: str, scope_key: str, user_name: str, persona_response: str) -> ContextSaveResponse:
        """
        Tells Fuli what this character actually said, closing out the
        memory buffer __call_Fuli__ opened for `user_key`. Must be
        called exactly once per turn (after the VL model produces its
        response) — Fuli refuses the next context request for the same
        user until this runs. Returns the full saved-memory record
        Fuli reports back.
        """
        return self.Emanator.save_context_memory(
            user_key=user_key,
            scope_key=scope_key,
            user_name=user_name,
            persona_response=persona_response,
        )

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
