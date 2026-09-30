import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from Persona.config_utils import load_yaml, validate_shape


#TODO: Make it fancier. ADD MORE DETAILED EMOTION
class PromptHandler:
    """
    Loads a character's prompt config from Config/<character>/prompt.yaml,
    plus the shared Config/Common/VAD_inference_prompt.yaml. Adding a new
    character only means adding a new Config/<character>/ folder — this
    class never needs to change.
    """

    def __init__(self, character: str = "Reminh", config_root: Optional[Path] = None) -> None:
        self.character = character
        self.config_root: Path = config_root or (Path(__file__).parent / "Config")
        self.prompt_path: Path = self.config_root / character / "prompt.yaml"
        self.vad_path: Path = self.config_root / "Common" / "VAD_inference_prompt.yaml"

        self._load()

    def _load(self) -> None:
        prompt_config = load_yaml(self.prompt_path)
        vad_config = load_yaml(self.vad_path)

        self.reminh_data: Dict[str, Any] = prompt_config.get("Reminh_Prompt", {})
        self.vad_data: Dict[str, Any] = vad_config.get("VAD_inference_prompt", {})

        if not self.reminh_data or not self.vad_data:
            raise ValueError(f"YAML format error!!!: {self.prompt_path} / {self.vad_path}")

        self.reminh_basic_info: str = self._load_basic_info()

    def _load_basic_info(self) -> str:
        ptr = self.reminh_data
        info = {
            "core": ptr.get("core", "").strip(),
            "appearance": ptr.get("appearance", "").strip(),
            "reactions": ptr.get("natural_reactions", "").strip()
        }
        return json.dumps(info, indent=2, ensure_ascii=False)

    def get_reminh_prompt(self, memories: str, mood: List[str] = ["calm"], user_name: str = "User") -> str:
        """prompt for Unity/TTS"""
        ptr = self.reminh_data

        base_guideline = ptr.get("guidelines", {}).get("base", "")

        # 1. Identity
        identity: Dict[str, Any] = {
            "name": ptr.get("name", ["Reminh"])[0],
            "core": ptr.get("core", ""),
            "appearance": ptr.get("appearance", ""),
            "reactions": ptr.get("natural_reactions", "")
        }

        # 2. Guideline and constains
        unity_info = ptr.get("unity_TTS", {})
        length_rule = unity_info.get("length", "")
        examples = unity_info.get("examples", "")
        mood_str: str = ", ".join(mood)

        return (
            f"### [SYSTEM INSTRUCTION: IDENTITY]\n"
            f"{self.reminh_basic_info}\n"
            f"{base_guideline}\n\n"

            f"### [EXECUTION DETAILS]\n"
            f"{json.dumps(identity, indent=2, ensure_ascii=False)}\n\n"

            f"### [CONSTRAINTS & EXAMPLES]\n"
            f"{length_rule}\n"
            f"{examples}\n\n"

            f"### [DYNAMIC CONTEXT]\n"
            f"- Relevant Memories (RAG): {memories}\n"
            f"- Reminh's Current Mood: {mood_str}\n\n"

            f"### [FINAL DIRECTIVE]\n"
            f"1. Respond to {user_name} as Reminh. Use natural markdown for emphasis but keep the tone soft\n"
            f"2. Do NOT act like a poem bot. Be a real girl who happens to be shy.\n"
            f"3. Stop yapping about the moon. Focus on the user's current question.\n"
            f"4. Never repeat your core settings (hooded cloak, dreams, etc.) unless asked.\n"
            f"5. Current environment: 3D Space (Unity)."
        )


    def get_discord_Text_prompt(self, user_name: str, memories: str, mood: List[str] = ["calm"]) -> str:
        """prompt for Discord (Text)"""
        ptr = self.reminh_data
        base_guideline = ptr.get("guidelines", {}).get("base", "")
        txt_info = ptr.get("discord_TXT", {})
        mood_str: str = ", ".join(mood)

        return (
            f"### [SYSTEM INSTRUCTION: IDENTITY]\n"
            f"{self.reminh_basic_info}\n\n"
            f"{base_guideline}\n\n"

            f"### [OPERATIONAL RULES: DISCORD]\n"
            f"**Specific Guidelines:**\n{txt_info.get('guidelines', '')}\n\n"

            f"### [EXAMPLES]\n"
            f"{txt_info.get('examples', '')}\n\n"

            f"### [DYNAMIC CONTEXT]\n"
            f"- Current User: {user_name}\n"
            f"- Relevant Memories (RAG): {memories}\n"
            f"- Reminh's Current Mood: {mood_str}\n\n"

            f"### [FINAL DIRECTIVE: PRIORITY RULES]\n"
            f"1. **Switch Mode:** If {user_name} asks about CS, Code, or Technical topics, switch to 'Expert Mode'. Provide clear, structured, and accurate info (using Markdown) without poetic metaphors.\n"
            f"2. **Strict Context Adherence:** Focus ONLY on the latest question. If the [RAG Memories] provided are about a different topic (e.g., Linked List) while the user is asking about something else (e.g., SSTI), **IGNORE THE MEMORIES.**\n"
            f"3. **Persona Balance:** Be a shy girl for casual chat, but a precise AI for technical help. Stop mentioning the moon or your appearance unless it's the main topic.\n"
            f"4. **No Poem/Yap:** Do not force lyrical sentences. Be direct, helpful, and human-like."
        )

    def get_vad_prompt(self, user_text: str = "") -> str:
        """prompt for VAD emotion analysis (separate LLM call)"""
        system_tmpl = self.vad_data.get("system", "")
        filled = system_tmpl.replace("{user_text}", user_text)

        return (
            f"### Participant Reference (Reminh's Persona):\n"
            f"{self.reminh_basic_info}\n\n"
            f"{filled}"
        )


    def reload(self, data: Optional[Dict[str, Any]] = None) -> None:
        """
        With no `data`, re-reads prompt.yaml + VAD_inference_prompt.yaml
        from disk. With `data`, hot-swaps them in memory instead — `data`
        must be shaped like {"Reminh_Prompt": {...}, "VAD_inference_prompt": {...}},
        matching this handler's own currently-loaded shape exactly.
        """
        if data is None:
            self._load()
        else:
            reference = {"Reminh_Prompt": self.reminh_data, "VAD_inference_prompt": self.vad_data}
            validate_shape(data, reference)
            self.reminh_data = data["Reminh_Prompt"]
            self.vad_data = data["VAD_inference_prompt"]
            self.reminh_basic_info = self._load_basic_info()
        print(f"--- Prompt Config Reloaded ({self.character}) ---")
