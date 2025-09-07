from pydantic import BaseModel, Field

class SummarizeTextRequest(BaseModel):
    text: str = Field(..., description="Text to summarize", min_length=1)