"""Adapter for VOZ post and comment crawler rows."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from schemas.models import UniversalSocialPost

logger = logging.getLogger(__name__)

_COMMENT_TIME_FORMATS = ("%b %d, %Y at %I:%M %p", "%Y-%m-%d %H:%M", "%Y-%m-%d")
_POST_TIME_FORMATS = ("%b %d, %Y", "%Y-%m-%d %H:%M", "%Y-%m-%d")
_NULL_VALUES = {"", "none", "null", "nan"}
_FALLBACK_CREATED_AT = 1_577_836_800  # 2020-01-01 UTC.


def _parse_time(value: object, formats: tuple[str, ...]) -> Optional[int]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return None


def _parse_count(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    digits = re.sub(r"[^0-9]", "", str(value))
    return int(digits) if digits else None


def _parse_reactions(value: object) -> int:
    if not isinstance(value, str):
        return 0
    return sum(int(count) for count in re.findall(r"\((\d+)\)", value))


def _make_comment_id(post_id: object, user_id: object, created_time: object) -> str:
    key = f"{post_id}_{user_id}_{created_time}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def _make_post_url(post_id: object) -> str:
    return f"https://voz.vn/t/{post_id}/"


def _as_int_if_possible(value: str) -> int | str:
    return int(value) if value.isdigit() else value


class VozAdapter:
    """Convert VOZ raw dictionaries into validated pandas rows."""

    def __init__(self, strict: bool = False):
        self.strict = strict

    def comments_to_df(self, raw_comments: list[dict]) -> pd.DataFrame:
        rows = self._convert_many(raw_comments, self._convert_comment, "VOZ comment")
        return pd.DataFrame(rows)

    def posts_to_df(self, raw_posts: list[dict]) -> pd.DataFrame:
        rows = self._convert_many(raw_posts, self._convert_post, "VOZ post")
        return pd.DataFrame(rows)

    def from_csv(
        self,
        comments_path: str,
        posts_path: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        comments = pd.read_csv(comments_path).to_dict("records")
        comments_df = self.comments_to_df(comments)

        posts_df = pd.DataFrame()
        if posts_path:
            posts = pd.read_csv(posts_path).to_dict("records")
            posts_df = self.posts_to_df(posts)
        return comments_df, posts_df

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

    def _convert_comment(self, raw: dict) -> dict:
        post_id = str(raw.get("id_post", "") or "")
        user_id = str(raw.get("id_user", "") or "")
        time_raw = raw.get("time", "")
        created_at = _parse_time(time_raw, _COMMENT_TIME_FORMATS)
        if created_at is None:
            raise ValueError(f"cannot parse time: {time_raw!r}")

        reaction_count = _parse_reactions(raw.get("reactions"))
        comment_id = _make_comment_id(post_id, user_id, time_raw)

        post = UniversalSocialPost(
            post_id=comment_id,
            source="voz",
            post_type="comment",
            author=str(raw.get("user", "") or ""),
            author_id=user_id,
            content=str(raw.get("comment", "") or ""),
            created_at=created_at,
            url=str(raw.get("url", "") or ""),
            parent_post_id=post_id,
            reaction_count=reaction_count,
        )
        return {
            "id_post": _as_int_if_possible(post_id),
            "id_user": _as_int_if_possible(user_id),
            "user": post.author,
            "time": time_raw,
            "comment": post.content,
            "url": post.url,
            "reactions": raw.get("reactions", ""),
            "comment_id": comment_id,
            "created_at": created_at,
            "reaction_count": reaction_count,
            "source": post.source,
            "post_type": post.post_type,
            "lang": post.lang,
        }

    def _convert_post(self, raw: dict) -> dict:
        post_id = str(raw.get("id_post", "") or "")
        title = str(raw.get("title", "") or "").strip()
        if post_id.lower() in _NULL_VALUES:
            raise ValueError(f"invalid id_post: {post_id!r}")
        if title.lower() in _NULL_VALUES:
            raise ValueError(f"invalid title: {title!r}")

        time_raw = raw.get("time_post", "")
        created_at = _parse_time(time_raw, _POST_TIME_FORMATS) or _FALLBACK_CREATED_AT
        author_id = str(raw.get("id_author", "") or "")
        author_name = str(raw.get("author_name", "") or "")
        category = str(raw.get("category", "") or "")
        subcategory = str(raw.get("subcategory", "") or "")
        view_count = _parse_count(raw.get("views_post"))
        comment_count = _parse_count(raw.get("replies_post"))

        post = UniversalSocialPost(
            post_id=post_id,
            source="voz",
            post_type="post",
            author=author_name,
            author_id=author_id,
            content=title,
            created_at=created_at,
            url=str(raw.get("url", "") or _make_post_url(post_id)),
            title=title,
            view_count=view_count,
            comment_count=comment_count,
            tags=[tag for tag in (category, subcategory) if tag],
        )
        return {
            "id_post": _as_int_if_possible(post_id),
            "title": post.title or "",
            "time_post": time_raw,
            "id_author": author_id,
            "author_name": author_name,
            "category": category,
            "subcategory": subcategory,
            "created_at": post.created_at,
            "view_count": view_count,
            "comment_count": comment_count,
            "source": post.source,
            "post_type": post.post_type,
            "tags": ",".join(post.tags),
        }
