"""
schemas/vatvo_adapter.py — Chuyển đổi raw data từ M1 VatVoStudio crawler → UniversalSocialPost

M1 gửi file articles.csv với các cột:
    post_id  : ID bài viết (số nguyên)
    article  : Tiêu đề bài viết
    author   : Tên tác giả
    time     : Thời gian đăng, dạng "April 3, 2026 at 8:44 am"
    content  : Nội dung toàn văn bài viết

Adapter làm các việc:
    1. Parse time string "April 3, 2026 at 8:44 am" → Unix timestamp (giây)
    2. Map đúng cột: article → title, content → content
    3. Validate qua UniversalSocialPost
    4. Trả về DataFrame sẵn sàng cho M4 sentiment pipeline

Cách dùng:
    from schemas.vatvo_adapter import VatVoAdapter

    adapter = VatVoAdapter()
    articles_df = adapter.from_csv("data/articles.csv")
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from pydantic import ValidationError

from schemas.models import UniversalSocialPost

logger = logging.getLogger(__name__)

# "April 3, 2026 at 8:44 am"
_TIME_FMT = "%B %d, %Y at %I:%M %p"

# URL gốc của VatVoStudio — dùng khi M1 không cung cấp url trực tiếp
_BASE_URL = "https://vatvo.studio/p/{post_id}/"


def _parse_time(time_str: str) -> Optional[int]:
    """
    Parse "April 3, 2026 at 8:44 am" → Unix timestamp (giây).
    Trả None nếu không parse được.
    """
    if not isinstance(time_str, str):
        return None
    s = time_str.strip()
    # Chuẩn hoá: "8:44 am" → "8:44 AM" (strptime cần chữ hoa)
    s = re.sub(r"\b(am|pm)\b", lambda m: m.group().upper(), s)
    try:
        dt = datetime.strptime(s, _TIME_FMT)
        return int(dt.replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        logger.warning("Không parse được time VatVo: %r", time_str)
        return None


class VatVoAdapter:
    """
    Adapter chuyển đổi M1 VatVoStudio articles.csv → DataFrame chuẩn cho M4 pipeline.

    Args:
        strict: nếu True, raise lỗi ngay khi có record invalid thay vì skip
    """

    def __init__(self, strict: bool = False):
        self.strict = strict

    def articles_to_df(self, raw_articles: list[dict]) -> pd.DataFrame:
        """
        Chuyển list raw article dict → DataFrame.

        Input dict keys : post_id, article, author, time, content
        Output columns  : post_id, title, author, time, content,
                          created_at, source, post_type, url
        """
        rows   = []
        n_skip = 0

        for raw in raw_articles:
            try:
                record = self._convert(raw)
            except (ValidationError, Exception) as e:
                n_skip += 1
                logger.warning("Skip article post_id=%s: %s", raw.get("post_id"), e)
                if self.strict:
                    raise
                continue
            rows.append(record)

        if n_skip:
            logger.warning("Bỏ qua %d/%d article không hợp lệ.", n_skip, len(raw_articles))

        if not rows:
            return pd.DataFrame(columns=[
                "post_id", "title", "author", "time", "content",
                "created_at", "source", "post_type", "url",
            ])

        return pd.DataFrame(rows)

    def from_csv(self, articles_path: str) -> pd.DataFrame:
        """
        Load articles.csv từ M1 → validate → trả articles_df.

        Args:
            articles_path: đường dẫn tới articles.csv

        Returns:
            DataFrame các bài viết đã validate, sẵn sàng cho M4 pipeline
        """
        logger.info("Đọc VatVo articles: %s", articles_path)
        raw = pd.read_csv(articles_path, encoding="utf-8").to_dict("records")
        return self.articles_to_df(raw)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _convert(self, raw: dict) -> dict:
        """Validate 1 raw article dict qua UniversalSocialPost → row dict."""
        post_id   = str(raw.get("post_id", ""))
        time_str  = str(raw.get("time", ""))
        title     = str(raw.get("article", "")).strip()
        content   = str(raw.get("content", "")).strip()
        author    = str(raw.get("author", "")).strip()
        url       = _BASE_URL.format(post_id=post_id)

        created_at = _parse_time(time_str)
        if created_at is None:
            raise ValueError(f"Không parse được time: {time_str!r}")

        # content là nội dung bài viết — dùng làm text chính cho NLP
        # title (article) lưu vào trường title
        post = UniversalSocialPost(
            post_id    = post_id,
            source     = "vatvo",       # cần thêm "vatvo" vào VALID_SOURCES
            post_type  = "article",
            author     = author or "unknown",
            content    = content or title,  # fallback sang title nếu content rỗng
            created_at = created_at,
            url        = url,
            title      = title,
        )

        return {
            "post_id":   post.post_id,
            "title":     post.title or "",
            "author":    post.author,
            "time":      time_str,
            "content":   post.content,
            "created_at": post.created_at,
            "source":    post.source,
            "post_type": post.post_type,
            "url":       post.url,
        }
