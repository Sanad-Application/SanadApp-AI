from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VDBInterface(ABC):

    @abstractmethod
    def connect(self):
        """Connect to the VDB."""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from the VDB."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the VDB is healthy and responsive."""
        pass


    @abstractmethod
    def create_collection(self, collection_name: str, embedding_size: int):
        """Create a collection in the VDB."""
        pass

    @abstractmethod
    def is_collection_exist(self, collection_name: str) -> bool:
        """Check if a collection exists in the VDB."""
        pass

    @abstractmethod
    def get_all_collections(self) -> List:
        """Get all collections in the VDB."""
        pass

    @abstractmethod
    def get_collection_info(self, collection_name: str):
        """Get information about a specific collection in the VDB."""
        pass

    @abstractmethod
    def insert_one(self, collection_name: str, vector: List[float], 
                   text: str, metadata: Dict[str, Any] = None, record_id: str = None):
        """Insert a single document into the VDB."""
        pass

    @abstractmethod
    def insert_many(self, collection_name: str, vectors: List[List[float]], 
                    texts: List[str], record_ids: List[str] = None, 
                    metadatas: List[Dict[str, Any]] = None, batch_size: int = 50):
        """Insert multiple documents into the VDB."""
        pass

    @abstractmethod
    def search(self, collection_name: str, query_vector: List[float], 
               top_k: int = 5, filter_conditions: Dict[str, Any] = None):
        """Search for similar documents in the VDB."""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str):
        """Delete a collection from the VDB."""
        pass

    @abstractmethod
    def delete_asset_chunks(self, collection_name: str, asset_id: str):
        """Delete all chunks associated with an asset from a collection."""
        pass