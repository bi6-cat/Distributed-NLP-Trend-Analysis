"""Adapter for VatVo article crawler rows."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from schemas.models import UniversalSocialPost

logger = logging.getLogger(__name__)

_TIME_FMT = "%B %d, %Y at %I:%M %p"
_BASE_URL = "https://vatvo.studio/p/{post_id}/"


def _parse_time(value: object) -> Optional[int]:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"\b(am|pm)\b", lambda m: m.group().upper(), value.strip())
    try:
        dt = datetime.strptime(normalized, _TIME_FMT)
        return int(dt.replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        return None


class VatVoAdapter:
    """Convert VatVo raw article dictionaries into validated pandas rows."""

    def __init__(self, strict: bool = False):
        self.strict = strict

    def articles_to_df(self, raw_articles: list[dict]) -> pd.DataFrame:
        rows = self._convert_many(raw_articles, self._convert, "VatVo article")
        return pd.DataFrame(rows)

    def _convert_many(self, records: list[dict], convert, label: str) -> list[dict]:
        rows = []
        for raw in records:
            try:
                rows.append(convert(raw))
            except Exception as exc:
                logger.warning("Skip invalid %s post_id=%s: %s", label, raw.get("post_id"), exc)
                if self.strict:
                    raise
        return rows

    def _convert(self, raw: dict) -> dict:
        post_id = str(raw.get("post_id", "") or "")
        title = str(raw.get("article", "") or "").strip()
        content = str(raw.get("content", "") or "").strip()
        author = str(raw.get("author", "") or "").strip() or "unknown"
        time_raw = raw.get("time", "")
        created_at = _parse_time(time_raw)

        if created_at is None:
            raise ValueError(f"cannot parse time: {time_raw!r}")

        post = UniversalSocialPost(
            post_id=post_id,
            source="vatvo",
            post_type="article",
            author=author,
            content=content or title,
            created_at=created_at,
            url=_BASE_URL.format(post_id=post_id),
            title=title,
        )
        return {
            "post_id": post.post_id,
            "title": post.title or "",
            "author": post.author,
            "time": time_raw,
            "content": post.content,
            "created_at": post.created_at,
            "source": post.source,
            "post_type": post.post_type,
            "url": post.url,
        }
