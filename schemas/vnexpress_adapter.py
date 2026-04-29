"""
schemas/vnexpress_adapter.py — Chuyển đổi raw data VnExpress → UniversalSocialPost

M1 gửi 2 file:
  POST (post_vnexpress.csv):
    id_post, link_post, post_time, post_content
    post_time dạng: "Thứ tư, 29/4/2026, 07:00 (GMT+7)"

  COMMENT (comment_vnexpress.csv):
    id_post, user_id, user_name, comment_content, comment_time, reaction_detail
    comment_time dạng: "07:28 29/04/2026"
    reaction_detail dạng: '{"Thích": 28}'
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
from pydantic import ValidationError

from schemas.models import UniversalSocialPost

logger = logging.getLogger(__name__)

_TZ_ICT = timezone(timedelta(hours=7))

# "29/4/2026, 07:00" — phần sau khi bỏ "Thứ X, "
_POST_TIME_RE  = re.compile(r"(\d{1,2}/\d{1,2}/\d{4}),\s*(\d{2}:\d{2})")
# "07:28 29/04/2026"
_CMT_TIME_FMT  = "%H:%M %d/%m/%Y"


def _parse_post_time(time_str: str) -> Optional[int]:
    """Parse "Thứ tư, 29/4/2026, 07:00 (GMT+7)" → Unix timestamp (giây)."""
    if not isinstance(time_str, str):
        return None
    m = _POST_TIME_RE.search(time_str)
    if not m:
        logger.warning("Không parse được post_time VnExpress: %r", time_str)
        return None
    try:
        dt = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%d/%m/%Y %H:%M")
        return int(dt.replace(tzinfo=_TZ_ICT).timestamp())
    except ValueError:
        logger.warning("Lỗi parse post_time VnExpress: %r", time_str)
        return None


def _parse_comment_time(time_str: str) -> Optional[int]:
    """Parse "07:28 29/04/2026" → Unix timestamp (giây)."""
    if not isinstance(time_str, str):
        return None
    try:
        dt = datetime.strptime(time_str.strip(), _CMT_TIME_FMT)
        return int(dt.replace(tzinfo=_TZ_ICT).timestamp())
    except ValueError:
        logger.warning("Không parse được comment_time VnExpress: %r", time_str)
        return None


def _parse_reactions(reaction_detail) -> int:
    """Parse '{"Thích": 28}' → 28. Trả 0 nếu lỗi."""
    if not reaction_detail or (isinstance(reaction_detail, float)):
        return 0
    try:
        d = json.loads(str(reaction_detail))
        return sum(int(v) for v in d.values() if str(v).isdigit())
    except Exception:
        return 0


def _make_comment_id(id_post, user_id, comment_time: str) -> str:
    key = f"vnexpress_{id_post}_{user_id}_{comment_time}"
    return "vne_" + hashlib.md5(key.encode()).hexdigest()[:12]


class VnExpressAdapter:
    """
    Adapter chuyển đổi M1 VnExpress raw CSV → DataFrame chuẩn cho M4 pipeline.

    Args:
        strict: nếu True, raise lỗi ngay khi có record invalid thay vì skip
    """

    def __init__(self, strict: bool = False):
        self.strict = strict

    def posts_to_df(self, raw_posts: list[dict]) -> pd.DataFrame:
        """
        Input keys : id_post, link_post, post_time, post_content
        Output     : DataFrame với post_id, source, author, title, body,
                     created_at, url, reaction_count, view_count, comment_count
        """
        rows, n_skip = [], 0
        for raw in raw_posts:
            try:
                rows.append(self._convert_post(raw))
            except Exception as e:
                n_skip += 1
                logger.warning("Skip post id_post=%s: %s", raw.get("id_post"), e)
                if self.strict:
                    raise
        if n_skip:
            logger.warning("Bỏ qua %d/%d post VnExpress không hợp lệ.", n_skip, len(raw_posts))
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def comments_to_df(self, raw_comments: list[dict]) -> pd.DataFrame:
        """
        Input keys : id_post, user_id, user_name, comment_content, comment_time, reaction_detail
        Output     : DataFrame với post_id, source, author, body,
                     created_at, parent_id, reaction_count
        """
        rows, n_skip = [], 0
        for raw in raw_comments:
            try:
                rows.append(self._convert_comment(raw))
            except Exception as e:
                n_skip += 1
                logger.warning("Skip comment id_post=%s user=%s: %s",
                               raw.get("id_post"), raw.get("user_id"), e)
                if self.strict:
                    raise
        if n_skip:
            logger.warning("Bỏ qua %d/%d comment VnExpress không hợp lệ.", n_skip, len(raw_comments))
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def from_csv(
        self,
        posts_path: str,
        comments_path: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load CSV → (posts_df, comments_df).
        comments_df là DataFrame rỗng nếu không truyền comments_path.
        """
        posts_df = self.posts_to_df(
            pd.read_csv(posts_path, encoding="utf-8").to_dict("records")
        )
        comments_df = pd.DataFrame()
        if comments_path:
            comments_df = self.comments_to_df(
                pd.read_csv(comments_path, encoding="utf-8").to_dict("records")
            )
        return posts_df, comments_df

    # ── Internal ──────────────────────────────────────────────────────────────

    def _convert_post(self, raw: dict) -> dict:
        id_post    = str(raw.get("id_post", ""))
        content    = str(raw.get("post_content", "") or "").strip()
        url        = str(raw.get("link_post", "") or "")
        time_str   = str(raw.get("post_time", "") or "")
        created_at = _parse_post_time(time_str)

        if not content:
            raise ValueError("post_content rỗng")
        if created_at is None:
            raise ValueError(f"Không parse được post_time: {time_str!r}")

        post = UniversalSocialPost(
            post_id    = f"vne_{id_post}",
            source     = "vnexpress",
            post_type  = "article",
            author     = "VnExpress",
            content    = content,
            created_at = created_at,
            url        = url,
        )
        return {
            "post_id":       post.post_id,
            "source":        post.source,
            "author":        post.author,
            "title":         None,
            "body":          post.content,
            "created_at":    created_at,
            "url":           url,
            "parent_id":     None,
            "reaction_count": 0,
            "view_count":    None,
            "comment_count": None,
        }

    def _convert_comment(self, raw: dict) -> dict:
        id_post    = str(raw.get("id_post", ""))
        user_id    = str(raw.get("user_id", ""))
        user_name  = str(raw.get("user_name", "") or "")
        content    = str(raw.get("comment_content", "") or "").strip()
        time_str   = str(raw.get("comment_time", "") or "")
        created_at = _parse_comment_time(time_str)
        reactions  = _parse_reactions(raw.get("reaction_detail"))
        comment_id = _make_comment_id(id_post, user_id, time_str)

        if not content:
            raise ValueError("comment_content rỗng")
        if created_at is None:
            raise ValueError(f"Không parse được comment_time: {time_str!r}")

        post = UniversalSocialPost(
            post_id        = comment_id,
            source         = "vnexpress",
            post_type      = "comment",
            author         = user_name or "unknown",
            content        = content,
            created_at     = created_at,
            url            = f"https://vnexpress.net/{id_post}",
            parent_post_id = f"vne_{id_post}",
            reaction_count = reactions,
        )
        return {
            "post_id":        post.post_id,
            "source":         post.source,
            "author":         post.author,
            "title":          None,
            "body":           post.content,
            "created_at":     created_at,
            "url":            post.url,
            "parent_id":      post.parent_post_id,
            "reaction_count": reactions,
            "view_count":     None,
            "comment_count":  None,
        }
