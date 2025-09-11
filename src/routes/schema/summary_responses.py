from pydantic import BaseModel
from typing import Optional

class SummaryResponse(BaseModel):
    success: bool
    message: str
    summary: Optional[str] = None