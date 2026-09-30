"""
Enigmata: a character's emotion-engine payload builder.

Loads the character's emotion "physics" policy from
Config/<character>/physics_weights.yaml (OCEAN traits + the stress/reward
and dynamics formula weights) and packages a raw VAD stimulus into the
payload FuliHandler ships off to the emotion engine on the Fuli C++
server. Enigmata never touches the network itself.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import config
from Persona.config_utils import load_yaml, validate_shape


class Enigmata:
    def __init__(self, character: str, config_root: Optional[Path] = None) -> None:
        self.character = character
        self.config_root: Path = config_root or (Path(__file__).parent / "Config")
        self.config_path: Path = self.config_root / character / "physics_weights.yaml"

        self._load()

    def _load(self) -> None:
        raw = load_yaml(self.config_path)
        self.ocean: Dict[str, float] = raw.get("OCEAN", {})
        self.analysis_weights: Dict[str, float] = raw.get("AnalysisWeights", {})
        self.formula_weights: Dict[str, Any] = raw.get("FormulaWeights", {})
        self.advanced_config: Dict[str, float] = raw.get("AdvancedConfig", {})

        if not self.ocean:
            raise ValueError(f"YAML format error!!!: {self.config_path}")

    @staticmethod
    def backend_flags() -> Dict[str, bool]:
        """
        Boolean flags telling Fuli's C++ server which backend to run
        emotion (VAD) inference on. Fuli does the actual inference itself
        — from config.emotion_inference, we only say which one to use and
        whether it should fall back to 5090 if openjev fails.
        """
        backend = config.emotion_inference.backend
        return {
            "use_openjev": backend == "openjev",
            "use_5090": backend == "5090",
            "fallback_to_5090": config.emotion_inference.fallback_to_5090,
        }

    def reload(self, data: Optional[Dict[str, Any]] = None) -> None:
        """
        With no `data`, re-reads physics_weights.yaml from disk. With
        `data`, hot-swaps it in memory instead — `data` must be shaped
        like {"OCEAN": {...}, "AnalysisWeights": {...}, "FormulaWeights": {...},
        "AdvancedConfig": {...}}, matching this handler's own currently-loaded
        shape exactly.
        """
        if data is None:
            self._load()
        else:
            reference = {
                "OCEAN": self.ocean,
                "AnalysisWeights": self.analysis_weights,
                "FormulaWeights": self.formula_weights,
                "AdvancedConfig": self.advanced_config,
            }
            validate_shape(data, reference)
            self.ocean = data["OCEAN"]
            self.analysis_weights = data["AnalysisWeights"]
            self.formula_weights = data["FormulaWeights"]
            self.advanced_config = data["AdvancedConfig"]
        print(f"--- Physics Config Reloaded ({self.character}) ---")
