from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class UploadResponse(BaseModel):
    success: bool
    message: str
    file_name: str
    asset_id: str
    file_size: int
    chunk_count: int
    embeddings_count: int
    inserted_count: int

class CollectionsResponse(BaseModel):
    success: bool
    message: str
    collections: List[str] 
    count: int


class CollectionInfoResponse(BaseModel):
    success: bool
    message: str
    collection_name: str
    vector_count: Optional[int] = None
    indexed_vector_count: Optional[int] = None
    points_count: Optional[int] = None
    payload_schema: Optional[Dict[str, Any]] = None

class DeleteAssetResponse(BaseModel):
    success: bool
    message: str
    asset_id: str

class DeleteCollectionResponse(BaseModel):
    success: bool
    message: str
    collection_name: str