"""Adapter for VnExpress post and comment crawler rows."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from schemas.models import UniversalSocialPost

logger = logging.getLogger(__name__)

_TZ_ICT = timezone(timedelta(hours=7))
_POST_TIME_RE = re.compile(r"(\d{1,2}/\d{1,2}/\d{4}),\s*(\d{2}:\d{2})")
_COMMENT_TIME_FMT = "%H:%M %d/%m/%Y"


def _normalize_numeric_id(value: object) -> str:
    try:
        return str(int(float(str(value))))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"invalid id: {value!r}")


def _parse_post_time(value: object) -> Optional[int]:
    if not isinstance(value, str):
        return None
    match = _POST_TIME_RE.search(value)
    if not match:
        return None
    try:
        dt = datetime.strptime(f"{match.group(1)} {match.group(2)}", "%d/%m/%Y %H:%M")
        return int(dt.replace(tzinfo=_TZ_ICT).timestamp())
    except ValueError:
        return None


def _parse_comment_time(value: object) -> Optional[int]:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.strptime(value.strip(), _COMMENT_TIME_FMT)
        return int(dt.replace(tzinfo=_TZ_ICT).timestamp())
    except ValueError:
        return None


def _parse_reactions(value: object) -> int:
    if not value or isinstance(value, float):
        return 0
    try:
        data = json.loads(str(value))
        return sum(int(v) for v in data.values() if str(v).isdigit())
    except Exception:
        return 0


def _make_comment_id(post_id: object, user_id: object, created_time: object) -> str:
    key = f"vnexpress_{post_id}_{user_id}_{created_time}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


class VnExpressAdapter:
    """Convert VnExpress raw dictionaries into validated pandas rows."""

    def __init__(self, strict: bool = False):
        self.strict = strict

    def posts_to_df(self, raw_posts: list[dict]) -> pd.DataFrame:
        rows = self._convert_many(raw_posts, self._convert_post, "VnExpress post")
        return pd.DataFrame(rows)

    def comments_to_df(self, raw_comments: list[dict]) -> pd.DataFrame:
        rows = self._convert_many(raw_comments, self._convert_comment, "VnExpress comment")
        return pd.DataFrame(rows)

    def _convert_many(self, records: list[dict], convert, label: str) -> list[dict]:
        rows = []
        for raw in records:
            try:
                rows.append(convert(raw))
            except Exception as exc:
                logger.warning("Skip invalid %s id_post=%s: %s", label, raw.get("id_post"), exc)
                if self.strict:
                    raise
        return rows

    def _convert_post(self, raw: dict) -> dict:
        post_id = _normalize_numeric_id(raw.get("id_post"))
        content = str(raw.get("post_content", "") or "").strip()
        url = str(raw.get("link_post", "") or "")
        created_at = _parse_post_time(raw.get("post_time", ""))

        if created_at is None:
            raise ValueError(f"cannot parse post_time: {raw.get('post_time')!r}")

        post = UniversalSocialPost(
            post_id=post_id,
            source="vnexpress",
            post_type="article",
            author="VnExpress",
            content=content,
            created_at=created_at,
            url=url,
        )
        return {
            "post_id": post.post_id,
            "source": post.source,
            "author": post.author,
            "title": None,
            "body": post.content,
            "created_at": created_at,
            "url": url,
            "parent_id": None,
            "reaction_count": 0,
            "view_count": None,
            "comment_count": None,
        }

    def _convert_comment(self, raw: dict) -> dict:
        parent_id = _normalize_numeric_id(raw.get("id_post"))
        user_id = str(raw.get("user_id", "") or "")
        user_name = str(raw.get("user_name", "") or "").strip() or "unknown"
        content = str(raw.get("comment_content", "") or "").strip()
        time_raw = raw.get("comment_time", "")
        created_at = _parse_comment_time(time_raw)
        reactions = _parse_reactions(raw.get("reaction_detail"))

        if created_at is None:
            raise ValueError(f"cannot parse comment_time: {time_raw!r}")

        post = UniversalSocialPost(
            post_id=_make_comment_id(parent_id, user_id, time_raw),
            source="vnexpress",
            post_type="comment",
            author=user_name,
            content=content,
            created_at=created_at,
            url=f"https://vnexpress.net/{parent_id}",
            parent_post_id=parent_id,
            reaction_count=reactions,
        )
        return {
            "post_id": post.post_id,
            "source": post.source,
            "author": post.author,
            "title": None,
            "body": post.content,
            "created_at": created_at,
            "url": post.url,
            "parent_id": post.parent_post_id,
            "reaction_count": reactions,
            "view_count": None,
            "comment_count": None,
        }
