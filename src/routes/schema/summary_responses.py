from pydantic import BaseModel
from typing import Optional

class SummaryResponse(BaseModel):
    success: bool
    message: str
    summary: Optional[str] = None
    asset_id: Optional[str] = None
    file_name: Optional[str] = None