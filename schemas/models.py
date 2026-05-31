"""Canonical validation model for crawler records.

Adapters use this model as a lightweight guard before returning pandas rows to
Spark cleaning. Spark should receive plain dictionaries/DataFrames, not
Pydantic objects.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


VALID_SOURCES = {"voz", "vnexpress", "youtube", "tinhte", "vatvo"}
VALID_POST_TYPES = {"post", "comment", "reply", "article", "video"}


class UniversalSocialPost(BaseModel):
    post_id: str
    source: str
    post_type: str
    author: str
    content: str
    created_at: int  # Unix timestamp in seconds.
    url: str

    parent_post_id: Optional[str] = None
    reaction_count: int = 0
    view_count: Optional[int] = None
    comment_count: Optional[int] = None

    title: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    author_id: Optional[str] = None
    lang: str = "vi"

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        if value not in VALID_SOURCES:
            raise ValueError(f"source={value!r} is not supported")
        return value

    @field_validator("post_type")
    @classmethod
    def validate_post_type(cls, value: str) -> str:
        if value not in VALID_POST_TYPES:
            raise ValueError(f"post_type={value!r} is not supported")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_unix_seconds(cls, value: int) -> int:
        if not (1_420_000_000 <= value <= 2_051_000_000):
            raise ValueError("created_at must be a Unix timestamp in seconds")
        return value

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("content must not be empty")
        return value

    def to_row(self) -> dict:
        row = self.model_dump()
        row["tags"] = ",".join(row["tags"]) if row["tags"] else ""
        return row
