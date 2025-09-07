from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class HealthResponse(BaseModel):
    initialized: bool
    message: str

class AnswerResponse(BaseModel):
    success: bool
    message: str
    answer: Optional[str] = None
    source_chunks: Optional[List[Dict[str, Any]]] = None

