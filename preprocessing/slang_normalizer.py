"""Vietnamese slang / teencode normalization."""

import json
import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)


class SlangNormalizer:
    """Replace whole-word slang terms with normalized Vietnamese text."""

    def __init__(self, dict_path: str = "data/slang_dict.json"):
        self.slang_dict = self._load_dict(dict_path)
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
            slang_dict = {k.lower(): v for k, v in slang_dict.items()}
            logger.info("[SlangNormalizer] Loaded %s slang entries from %s", len(slang_dict), path)
            return slang_dict
        except FileNotFoundError:
            logger.warning("[SlangNormalizer] Slang dict not found: %s", path)
            return {}
        except json.JSONDecodeError as e:
            logger.warning("[SlangNormalizer] Invalid JSON in %s: %s", path, e)
            return {}
