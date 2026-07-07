import re
import json
from typing import Dict, Optional, Any
from pathlib import Path
from faiss import VERSION_STRING
from .RAG.Fuli_v2 import *
import delta_ego_core
import yaml
from .PromptHandler import PromptHandler


class Reminh:
    def __init__(self):
        self.last_emotion: Dict[str, Any]
        self.last_emotion_terms: List[str]
        self.last_user_in: str

        # Prepares values for variables
        current_dir = Path(__file__).parent.resolve()
        config_path = current_dir / "Reminh_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        mem_config = config.get("memory", {})
        emo_config = config.get("personality", {})
        # check if it is empty or not
        if not mem_config or not emo_config:
            raise Exception(
                "No mem_config or emo_config!! Check Reminh_config.yaml or logic in Reminh.py!!!"
            )

        # For MemoryHandler
        self.yaml_mem_queue_len: int = mem_config.get("queue_len", 10)
        self.yaml_top_k: int = mem_config.get("top_k", 5)
        self.yaml_mem_update_cnt: int = mem_config.get("update_cnt", 10)
        # For EmotionModule
        personal_trait = emo_config.get("ocean", {})
        default_VAD_Area = emo_config.get("vad_base", {})

        if not personal_trait or not default_VAD_Area:
            raise Exception("No Personality DATA in Reminh_config.yaml")

            # Personal traits
        self.yaml_O: float = personal_trait.get("O", 0.8)
        self.yaml_C: float = personal_trait.get("C", 0.7)
        self.yaml_E: float = personal_trait.get("E", 0.4)
        self.yaml_A: float = personal_trait.get("A", 0.9)
        self.yaml_N: float = personal_trait.get("N", 0.2)

        # def VAD Area
        self.yaml_def_V: float = float(default_VAD_Area.get("V", 0.0))
        self.yaml_def_A: float = float(default_VAD_Area.get("A", 0.0))
        self.yaml_def_D: float = float(default_VAD_Area.get("D", 0.0))
        self.yaml_def_radius: float = float(default_VAD_Area.get("radius", 0.1))

        # Memory--------------------------------------------------------------------
        self.MemoryHandler: Fuli = Fuli(
            mem_queue_len=self.yaml_mem_queue_len,
            top_k=self.yaml_top_k,
            mem_update_cnt=self.yaml_mem_update_cnt,
        )
        # --------------------------------------------------------------------------

        # Emotion-------------------------------------------------------------------
        self.ego_physics_config = str(current_dir / "Reminh_physics.yaml")
        self.EmotionModule = delta_ego_core.deltaEGO(
            self.ego_physics_config,
            self.yaml_def_V,
            self.yaml_def_A,
            self.yaml_def_D,
            self.yaml_def_radius,
        )

        if self.EmotionModule.load_vad_db("/home/user/work/Reminh/VAD.json"):
            print("VAD Database loaded successfully!")
        else:
            print("Failed to load VAD.json. Check the file path.")

        # --------------------------------------------------------------------------

        # Prompt--------------------------------------------------------------------
        self.PromptModule: PromptHandler = PromptHandler()
        # --------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------

    def reload_physics_config(self) -> bool:
        """Reloads the YAML physics settings for the C++ engine in real-time."""
        return self.EmotionModule.reload_config()

    def get_VAD_prompt(self, user_text: str = "") -> str:
        reval = self.PromptModule.get_vad_prompt(user_text=user_text)
        return reval

    def get_Reminh_prompt(
        self,
        VAD_result_raw: str,
        user_input: str,
        source: str = "unity",
        user_name: str = "User",
    ) -> str:
        self.last_user_in = user_input

        parsed_VAD: Dict[str, float] = self._parse_vad_json(raw_string=VAD_result_raw)

        # Emotion
        raw_emotion_result: str = self.EmotionModule.process_stimulus(
            parsed_VAD.get("Valence", 0.0),
            parsed_VAD.get("Arousal", 0.0),
            parsed_VAD.get("Dominance", 0.0),
        )

        try:
            emotion_result = json.loads(raw_emotion_result)
        except Exception as e:
            print(f"[Critical] Emotion JSON Parse Error: {e}")
            emotion_result = {}

        self.last_emotion = emotion_result

        # debug
        print(f"DEBUG - RAW JSON : {emotion_result}")
        print(f"DEBUG - Emotion Keys: {emotion_result.keys()}")

        # TODO: Make it multiple results like 3
        if "emotion_term" not in emotion_result:
            print("[System Warning] Emotion term missing! Falling back to 'calm'.")
        emotion_terms: str = emotion_result.get("emotion_term", "[calm]")
        self.last_emotion_terms = [emotion_terms]
        # RAG
        memories_result: str = self.MemoryHandler.retrieve(user_input, None)

        # making prompt
        if source.lower() == "unity":
            final_prompt = self.PromptModule.get_reminh_prompt(
                user_name=user_name, 
                memories=memories_result,
                mood=[emotion_terms]
            )
        elif source.lower() == "discord_txt":
            # DISCORD
            final_prompt = self.PromptModule.get_discord_Text_prompt(
                user_name=user_name, 
                memories=memories_result, 
                mood=[emotion_terms]
            )

        return final_prompt

    def set_Reminh_memory(self, User_Name: str, AI_output: str) -> None:
        self.MemoryHandler.add_memory(
            user_in=self.last_user_in,
            user_name=User_Name,
            AI_response=AI_output,
            AI_status=str(self.last_emotion_terms),
            deltaEGO_analysis=self.last_emotion,
        )
        return

    def get_Reminh_last_emotion(self) -> Dict[str, Any]:
        return self.last_emotion

    def _parse_vad_json(self, raw_string: str) -> Dict[str, float]:
        try:
            if "```" in raw_string:
                pattern = r"```(?:json)?\s*({.*?})\s*```"
                match = re.search(pattern, raw_string, re.DOTALL)
                if match:
                    raw_string = match.group(1)
                else:
                    raw_string = raw_string.replace("```json", "").replace("```", "")

            raw_string = raw_string.strip()

            # If the model mixed narrative + JSON, extract just the JSON
            if not raw_string.startswith("{"):
                match = re.search(r'\{[^{}]*\}', raw_string)

                if match:
                    raw_string = match.group()

            return json.loads(raw_string)

        except (json.JSONDecodeError, AttributeError) as e:
            print(f"[System] JSON Parsing Error: {e}")
            print(f"[System] Raw String was: {raw_string}")
            return {"Valence": 0.0, "Arousal": 0.0, "Dominance": 0.0}

    def get_info(self) -> None:
        pass
