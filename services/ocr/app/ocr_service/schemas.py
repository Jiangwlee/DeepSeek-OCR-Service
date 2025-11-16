from __future__ import annotations

import uuid
from typing import List, Literal, Optional

from fastapi import Form
from pydantic import BaseModel, Field, HttpUrl


class DocumentOptions(BaseModel):
    output_format: Literal["markdown", "plain_text"] = "markdown"
    prompt: Optional[str] = None
    store_result: bool = False
    store_bucket: Optional[str] = None
    store_object_name: Optional[str] = None
    provider: Literal["deepseek", "paddle"] = "deepseek"

    @classmethod
    def as_form(
        cls,
        output_format: Literal["markdown", "plain_text"] = Form("markdown"),
        prompt: Optional[str] = Form(None),
        store_result: bool = Form(False),
        store_bucket: Optional[str] = Form(None),
        store_object_name: Optional[str] = Form(None),
        provider: Literal["deepseek", "paddle"] = Form("deepseek"),
    ) -> "DocumentOptions":
        return cls(
            output_format=output_format,
            prompt=prompt,
            store_result=store_result,
            store_bucket=store_bucket,
            store_object_name=store_object_name,
            provider=provider,
        )


class StorageSource(BaseModel):
    bucket: str
    object_name: str


class URLSource(BaseModel):
    url: HttpUrl


class StorageOCRRequest(DocumentOptions):
    source: StorageSource


class URLOCRRequest(DocumentOptions):
    source: URLSource


class PageResult(BaseModel):
    index: int
    text: str


class DocumentResult(BaseModel):
    document_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    output_format: Literal["markdown", "plain_text"]
    prompt_used: str
    total_pages: int
    pages: List[PageResult]
    combined_text: str
    stored_bucket: Optional[str] = None
    stored_object_name: Optional[str] = None
