from typing import Optional

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
