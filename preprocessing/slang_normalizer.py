"""
Chuẩn hóa teencode / slang tiếng Việt → tiếng Việt chuẩn.
Ví dụ: "ko" → "không", "dc" → "được", "nc" → "nói chuyện"

Member 4 — NLP Engineer
"""

import json
import re
from typing import Dict


class SlangNormalizer:
    """
    Normalize Vietnamese teencode/slang.

    Slang dict format (JSON):
    {
        "ko": "không",
        "dc": "được",
        "ntn": "như thế nào",
        "trc": "trước",
        "nc": "nói chuyện",
        "đc": "được",
        "bt": "bình thường",
        "mk": "mình",
        "bik": "biết",
        "thik": "thích",
        "lm": "làm",
        ...
    }
    """

    def __init__(self, dict_path: str = "data/slang_dict.json"):
        self.slang_dict = self._load_dict(dict_path)
        # Build regex: match whole word only, sorted dài → ngắn (greedy)
        if self.slang_dict:
            sorted_keys = sorted(self.slang_dict.keys(), key=len, reverse=True)
            escaped_keys = [re.escape(k) for k in sorted_keys]
            self._pattern = re.compile(
                r"\b(" + "|".join(escaped_keys) + r")\b",
                flags=re.IGNORECASE,
            )
        else:
            self._pattern = None

    def normalize(self, text: str) -> str:
        """Thay thế slang/teencode bằng tiếng Việt chuẩn."""
        if not self._pattern or not text:
            return text

        def _replace(match):
            word = match.group(0).lower()
            return self.slang_dict.get(word, word)

        return self._pattern.sub(_replace, text)

    def _load_dict(self, path: str) -> Dict[str, str]:
        """Load slang dictionary từ JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                slang_dict = json.load(f)
            # Normalize keys về lowercase
            slang_dict = {k.lower(): v for k, v in slang_dict.items()}
            print(f"[SlangNormalizer] Loaded {len(slang_dict)} slang entries from {path}")
            return slang_dict
        except FileNotFoundError:
            print(f"[SlangNormalizer] WARNING: Slang dict not found: {path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"[SlangNormalizer] ERROR: Invalid JSON in {path}: {e}")
            return {}