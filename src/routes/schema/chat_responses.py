from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class HealthResponse(BaseModel):
    initialized: bool
    message: str

class ChatResponse(BaseModel):
    success: bool
    message: str
    answer: Optional[str] = None

