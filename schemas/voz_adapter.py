"""
schemas/voz_adapter.py — Chuyển đổi raw data từ M1 VOZ crawler → UniversalSocialPost

M1 gửi 2 loại dict:

  POST:
    {"id_post", "title", "time_post", "replies_post", "views_post",
     "id_author", "author_name", "category", "subcategory"}

  COMMENT:
    {"id_post", "id_user", "user", "time", "comment", "url", "reactions"}

Adapter làm các việc:
  1. Parse time string → Unix timestamp (giây)
  2. Parse "Replies\\n28" → 28,  "Views\\n4,050" → 4050
  3. Parse "Ưng (3) | Haha (1)" → 4
  4. Tạo unique comment_id từ (id_post, id_user, time) vì M1 không có
  5. Validate qua UniversalSocialPost — lỗi được log, không crash toàn batch
  6. Trả về DataFrame sẵn sàng cho M4 pipeline

Cách dùng:
    from schemas.voz_adapter import VozAdapter

    adapter = VozAdapter()
    comments_df = adapter.comments_to_df(raw_comments_list)
    posts_df    = adapter.posts_to_df(raw_posts_list)

    # Hoặc từ CSV:
    comments_df, posts_df = adapter.from_csv(
        comments_path="data/voz_comments.csv",
        posts_path="data/voz_posts.csv",
    )
"""

from __future__ import annotations

import hashlib
import re
import logging
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
from pydantic import ValidationError

from schemas.models import UniversalSocialPost

logger = logging.getLogger(__name__)

_COMMENT_TIME_FMT = "%b %d, %Y at %I:%M %p"   # "Feb 23, 2026 at 4:00 PM"
_POST_TIME_FMT    = "%b %d, %Y"                 # "Feb 13, 2026"

# Format ISO từ CSV crawler thực tế
_ISO_DATETIME_FMT = "%Y-%m-%d %H:%M"            # "2025-01-07 15:23"
_ISO_DATE_FMT     = "%Y-%m-%d"                   # "2025-01-07"


def _parse_time(time_str: str, fmt: str) -> Optional[int]:
    """
    Parse chuỗi thời gian → Unix timestamp (giây). Trả None nếu lỗi.
    Thử fmt chính trước, fallback sang ISO format nếu thất bại.
    """
    if not isinstance(time_str, str):
        return None
    for f in [fmt, _ISO_DATETIME_FMT, _ISO_DATE_FMT]:
        try:
            dt = datetime.strptime(time_str.strip(), f)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    logger.warning("Không parse được time: %r (fmt=%s)", time_str, fmt)
    return None


def _parse_count(raw) -> Optional[int]:
    """
    Parse các chuỗi kiểu "Replies\\n28" hoặc "Views\\n4,050" → int.
    Trả None nếu không parse được.
    """
    if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
        return None
    s = str(raw)
    digits = re.sub(r"[^0-9]", "", s)   
    return int(digits) if digits else None


def _parse_reactions(raw) -> int:
    """Parse "Ưng (3) | Haha (1)" → 4. Trả 0 nếu NaN."""
    if not isinstance(raw, str):
        return 0
    counts = re.findall(r"\((\d+)\)", raw)
    return sum(int(c) for c in counts)


def _make_comment_id(id_post, id_user, time_str: str) -> str:
    """
    Tạo unique comment ID vì M1 không cung cấp.
    Dùng hash MD5 ngắn của (id_post, id_user, time).
    """
    key = f"{id_post}_{id_user}_{time_str}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def _make_post_url(id_post) -> str:
    """Construct VOZ post URL từ id."""
    return f"https://voz.vn/t/{id_post}/"


class VozAdapter:
    """
    Adapter chuyển đổi M1 VOZ raw data → DataFrame chuẩn cho M4 pipeline.

    Thiết kế:
      - Validate từng record qua UniversalSocialPost
      - Record lỗi: log warning + skip (không crash toàn batch)
      - Output DataFrame giữ nguyên column names mà M4 hiện đang dùng
        để không cần thay đổi isolation_forest / rolling_threshold

    Args:
        strict: nếu True, raise lỗi ngay khi có record invalid thay vì skip
    """

    def __init__(self, strict: bool = False):
        self.strict = strict

    # ── Public API ────────────────────────────────────────────────────────────

    def comments_to_df(self, raw_comments: list[dict]) -> pd.DataFrame:
        """
        Chuyển list raw comment dict (từ M1) → DataFrame cho M4.

        Input dict keys: id_post, id_user, user, time, comment, url, reactions

        Output columns (giữ tương thích với M4 hiện tại):
            id_post, id_user, user, time, comment, url, reactions,
            comment_id, created_at, reaction_count
        """
        rows = []
        n_skip = 0

        for raw in raw_comments:
            try:
                record = self._convert_comment(raw)
            except (ValidationError, ValueError) as e:
                n_skip += 1
                logger.warning("Skip comment id_post=%s id_user=%s: %s",
                               raw.get("id_post"), raw.get("id_user"), e)
                if self.strict:
                    raise
                continue
            rows.append(record)

        if n_skip:
            logger.warning("Bỏ qua %d/%d comment không hợp lệ.", n_skip, len(raw_comments))

        if not rows:
            return pd.DataFrame(columns=[
                "id_post", "id_user", "user", "time", "comment", "url",
                "reactions", "comment_id", "created_at", "reaction_count",
            ])

        return pd.DataFrame(rows)

    def posts_to_df(self, raw_posts: list[dict]) -> pd.DataFrame:
        """
        Chuyển list raw post dict (từ M1) → DataFrame cho M4.

        Input dict keys: id_post, title, time_post, replies_post,
                         views_post, id_author, author_name,
                         category, subcategory

        Output columns:
            id_post, title, time_post, id_author, author_name,
            category, subcategory, created_at, view_count, comment_count, url
        """
        rows = []
        n_skip = 0

        for raw in raw_posts:
            try:
                record = self._convert_post(raw)
            except (ValidationError, ValueError) as e:
                n_skip += 1
                logger.warning("Skip post id_post=%s: %s", raw.get("id_post"), e)
                if self.strict:
                    raise
                continue
            rows.append(record)

        if n_skip:
            logger.warning("Bỏ qua %d/%d post không hợp lệ.", n_skip, len(raw_posts))

        if not rows:
            return pd.DataFrame(columns=[
                "id_post", "title", "time_post", "id_author", "author_name",
                "category", "subcategory", "created_at", "view_count",
                "comment_count", "url",
            ])

        return pd.DataFrame(rows)

    def from_csv(
        self,
        comments_path: str,
        posts_path: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load CSV từ M1 → validate → trả (comments_df, posts_df).

        Args:
            comments_path: đường dẫn tới voz_comments.csv
            posts_path:    đường dẫn tới voz_posts.csv (tuỳ chọn)

        Returns:
            (comments_df, posts_df) — posts_df là DataFrame rỗng nếu không truyền
        """
        logger.info("Đọc comments: %s", comments_path)
        comments_raw = pd.read_csv(comments_path).to_dict("records")
        comments_df  = self.comments_to_df(comments_raw)

        posts_df = pd.DataFrame()
        if posts_path:
            logger.info("Đọc posts: %s", posts_path)
            posts_raw = pd.read_csv(posts_path).to_dict("records")
            posts_df  = self.posts_to_df(posts_raw)

        return comments_df, posts_df


    def _convert_comment(self, raw: dict) -> dict:
        """Validate 1 raw comment dict qua UniversalSocialPost → row dict."""
        id_post   = str(raw.get("id_post", ""))
        id_user   = str(raw.get("id_user", ""))
        time_str  = raw.get("time", "")
        created_at = _parse_time(time_str, _COMMENT_TIME_FMT)

        if created_at is None:
            raise ValidationError.from_exception_data(
                title="VozAdapter",
                input_type="python",
                line_errors=[{
                    "type": "value_error",
                    "loc": ("created_at",),
                    "msg": f"Không parse được time: {time_str!r}",
                    "input": time_str,
                    "ctx": {"error": ValueError(f"bad time: {time_str!r}")},
                }],
            )

        reaction_count = _parse_reactions(raw.get("reactions"))
        comment_id     = _make_comment_id(id_post, id_user, time_str)

        post = UniversalSocialPost(
            post_id        = comment_id,
            source         = "voz",
            post_type      = "comment",
            author         = str(raw.get("user", "")),
            author_id      = id_user,
            content        = str(raw.get("comment", "")),
            created_at     = created_at,
            url            = str(raw.get("url", "")),
            parent_post_id = id_post,
            reaction_count = reaction_count,
        )
        return {
            "id_post":        int(id_post) if id_post.isdigit() else id_post,
            "id_user":        int(id_user) if id_user.isdigit() else id_user,
            "user":           post.author,
            "time":           time_str,        
            "comment":        post.content,
            "url":            post.url,
            "reactions":      raw.get("reactions", ""),
            "comment_id":     comment_id,      
            "created_at":     created_at,     
            "reaction_count": reaction_count, 
            "source":         post.source,
            "post_type":      post.post_type,
            "lang":           post.lang,
        }

    def _convert_post(self, raw: dict) -> dict:
        """Validate 1 raw post dict qua UniversalSocialPost → row dict."""
        id_post    = str(raw.get("id_post", "") or "")

        # ── Lọc rác: bỏ qua các bản ghi crawler ghi sai id_post hoặc title = "None" ──
        _null_vals = {"", "none", "null", "nan"}
        if id_post.lower() in _null_vals:
            raise ValueError(f"id_post không hợp lệ: {id_post!r}")
        title_raw = str(raw.get("title", "") or "")
        if title_raw.lower() in _null_vals:
            raise ValueError(f"title không hợp lệ: {title_raw!r}")

        time_str   = raw.get("time_post", "")
        created_at = _parse_time(time_str, _POST_TIME_FMT)

        view_count    = _parse_count(raw.get("views_post"))
        comment_count = _parse_count(raw.get("replies_post"))
        id_author     = str(raw.get("id_author", ""))
        author_name   = str(raw.get("author_name", ""))
        category      = str(raw.get("category", ""))
        subcategory   = str(raw.get("subcategory", ""))
        url           = str(raw.get("url", _make_post_url(id_post)))

        post = UniversalSocialPost(
            post_id       = id_post,
            source        = "voz",
            post_type     = "post",
            author        = author_name,
            author_id     = id_author,
            content       = str(raw.get("title", "")).strip() or "(no title)",
            created_at    = created_at or 1_577_836_800,  # fallback: 2020-01-01
            url           = url,
            title         = str(raw.get("title", "")),
            view_count    = view_count,
            comment_count = comment_count,
            tags          = [t for t in [category, subcategory] if t],
        )

        return {
            "id_post":       int(id_post) if id_post.isdigit() else id_post,
            "title":         post.title or "",
            "time_post":     time_str,
            "id_author":     id_author,
            "author_name":   author_name,
            "category":      category,
            "subcategory":   subcategory,
            "created_at":    post.created_at,
            "view_count":    view_count,
            "comment_count": comment_count,
            "source":        post.source,
            "post_type":     post.post_type,
            "tags":          ",".join(post.tags),
        }
