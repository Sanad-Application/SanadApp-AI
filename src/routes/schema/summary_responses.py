from pydantic import BaseModel
from typing import Optional

class SummaryResponse(BaseModel):
    success: bool
    message: str
    summary: Optional[str] = None
    original_text_length: Optional[int] = None
    summary_length: Optional[int] = None
    asset_id: Optional[str] = None
    file_name: Optional[str] = None