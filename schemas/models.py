"""
schemas/models.py — Schema chuẩn hoá dữ liệu raw từ các nguồn crawl.

UniversalSocialPost là contract giữa M1 (crawler) và các thành viên còn lại:
  - M4 dùng để nhận và validate data trước khi chạy NLP pipeline
  - M3 dùng để nhận data trước khi chạy topic modeling
  - M5 dùng để đọc data cho dashboard

Quan trọng:
  - Pydantic chỉ dùng ở tầng INGESTION để validate
  - Trước khi đưa vào Spark/pandas pipeline, luôn gọi .to_row()
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, field_validator, model_validator


VALID_SOURCES    = {"voz", "vnexpress", "youtube", "tinhte", "vatvo"}
VALID_POST_TYPES = {"post", "comment", "reply", "article", "video"}


class UniversalSocialPost(BaseModel):
    # CORE 
    post_id:        str   # comment_id / thread_id / video_id — unique per record
    source:         str   # "voz" | "vnexpress" | "youtube" | "tinhte"
    post_type:      str   # "post" | "comment" | "reply" | "article" | "video"
    author:         str   # username / authorDisplayName
    content:        str   # nội dung chính (comment / article body / video desc)
    created_at:     int   # Unix timestamp — ĐƠN VỊ: giây (không phải ms)
    url:            str   # link gốc để truy vết

    # THREADING 
    parent_post_id: Optional[str] = None  # thread_id nếu đây là comment

    # ENGAGEMENT (dùng cho IsolationForest feature: engagement_score) ──────
    reaction_count: int            = 0     # likes / Ưng / hearts — đã parse sang int
    view_count:     Optional[int]  = None  # views (YouTube, VnExpress)
    comment_count:  Optional[int]  = None  # số replies của bài post

    # METADATA 
    title:          Optional[str]  = None  # tiêu đề (VnExpress article, YouTube video)
    tags:           List[str]      = []    # category / subcategory / hashtags
    author_id:      Optional[str]  = None  # id_user / id_author / channelId
    lang:           str            = "vi"

    # ── VALIDATORS 
    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        if v not in VALID_SOURCES:
            raise ValueError(f"source='{v}' không hợp lệ. Chọn từ: {VALID_SOURCES}")
        return v

    @field_validator("post_type")
    @classmethod
    def validate_post_type(cls, v: str) -> str:
        if v not in VALID_POST_TYPES:
            raise ValueError(f"post_type='{v}' không hợp lệ. Chọn từ: {VALID_POST_TYPES}")
        return v

    @field_validator("created_at")
    @classmethod
    def validate_unix_seconds(cls, v: int) -> int:
        # 2015-01-01 = 1_420_000_000  |  2035-01-01 = 2_051_000_000
        if not (1_420_000_000 <= v <= 2_051_000_000):
            raise ValueError(
                f"created_at={v} nằm ngoài khoảng hợp lệ. "
                "Đảm bảo đơn vị là GIÂY (không phải milliseconds)."
            )
        return v

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("content không được rỗng")
        return v.strip()

    # ── Serialization ─────────────────────────────────────────────────────────
    def to_row(self) -> dict:
        """
        Convert sang dict để đưa vào pandas DataFrame hoặc Spark RDD.

        QUAN TRỌNG: Không dùng Pydantic object trực tiếp trong Spark —
        Spark không serialize được Pydantic v2 object.
        """
        d = self.model_dump()
        d["tags"] = ",".join(d["tags"]) if d["tags"] else ""
        return d
