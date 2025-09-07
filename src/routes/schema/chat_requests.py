from pydantic import BaseModel, Field
from typing import Optional


class AnswerRequest(BaseModel):
    query: str = Field(..., description="Search query text")
