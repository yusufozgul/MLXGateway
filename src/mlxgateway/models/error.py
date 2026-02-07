from typing import Optional

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """OpenAI-compatible error detail"""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """OpenAI-compatible error response"""
    error: ErrorDetail
