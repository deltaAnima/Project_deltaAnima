"""
Shared yaml loading + shape-validation helpers for every Config-driven
handler (PromptHandler, Enigmata, Remembrance).
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot find file!!!: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML Parsing error in {path}: {e}")


def validate_shape(new: Any, reference: Any, path: str = "root") -> None:
    """
    Recursively checks that `new` has the same dict-key shape as
    `reference` (the handler's currently-loaded yaml). Only checks
    structure — missing, extra, or wrongly-nested keys — never leaf
    values, since a hot reload is allowed to change values, just not
    silently change the schema underneath the handler.
    """
    if isinstance(reference, dict):
        if not isinstance(new, dict):
            raise ValueError(f"{path}: expected a mapping, got {type(new).__name__}")
        missing = reference.keys() - new.keys()
        extra = new.keys() - reference.keys()
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        if extra:
            raise ValueError(f"{path}: unexpected keys {sorted(extra)}")
        for key in reference:
            validate_shape(new[key], reference[key], path=f"{path}.{key}")
    elif isinstance(reference, list):
        if not isinstance(new, list):
            raise ValueError(f"{path}: expected a list, got {type(new).__name__}")
        # element count/content can legitimately change; only the container type is checked
