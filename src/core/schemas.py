from pydantic import BaseModel
from typing import Optional, List


class Message(BaseModel):
    role: str
    content: str

class CompletionRequest(BaseModel):
    messages: List[Message]
    model: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
