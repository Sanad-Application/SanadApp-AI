from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException
from ..VDBInterface import VDBInterface
import logging
from typing import List, Dict, Any, Optional
import uuid
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class QdrantProvider(VDBInterface):
    def __init__(self, host: str = "localhost", port: int = 6333, 
                 grpc_port: int = 6334, distance_method: str = "cosine"):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.client = None
        
        # Set distance metric
        if distance_method == "cosine":
            self.distance_metric = models.Distance.COSINE
        elif distance_method == "dot":
            self.distance_metric = models.Distance.DOT
        else:
            self.distance_metric = models.Distance.COSINE
            logger.warning(f"Unknown distance method '{distance_method}', using cosine")

    async def connect(self):
        """Connect to Qdrant."""
        try:
            self.client = AsyncQdrantClient(
                host=self.host,
                port=self.port,
                prefer_grpc=False,  # Use HTTP instead of gRPC
            )

            # Test connection
            await self.client.get_collections()
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {self.host}:{self.port} -> {e}")
            raise ConnectionError(f"Health check failed: {e}") from e

    async def disconnect(self):
        if self.client:
            try:
                await self.client.close()
                logger.info("Disconnected from Qdrant")
            except Exception as e:
                logger.error(f"Error disconnecting from Qdrant: {e}")
            finally:
                self.client = None

    async def health_check(self) -> bool:
        """Check if Qdrant is reachable and healthy."""
        try:
            if not self.client:
                logger.warning("Qdrant client not initialized")
                return False
            await self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    @asynccontextmanager
    async def _ensure_connection(self):
        """to ensure connection is active."""
        if not self.client or not await self.health_check():
            await self.connect()
        try:
            yield
        except ResponseHandlingException as e:
            logger.error(f"Qdrant response error: {e}")
            if e.__cause__ and isinstance(e.__cause__, (ConnectionError, TimeoutError)):
                await self.connect()
            raise

    async def create_collection(self, collection_name: str, embedding_size: int):
        """Create a collection with optimized settings for production."""
        try:        
            async with self._ensure_connection():
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_size,
                        distance=self.distance_metric,
                        on_disk=True  # Store vectors on disk
                    ))

            logger.info(f"Collection '{collection_name}' created with optimized settings")
        except Exception as e:
            logger.error(f"Error creating collection '{collection_name}': {e}")
            raise
    
    async def is_collection_exist(self, collection_name: str) -> bool:
        try:
            return await self.client.collection_exists(collection_name)
        except Exception as e:
            logger.error(f"Error checking if collection exists '{collection_name}': {e}")
            raise

    async def get_all_collections(self) -> List:
        """Get all collections with the prefix."""
        async with self._ensure_connection():
            all_collections = await self.client.get_collections()
            return [collection.name for collection in all_collections.collections]

    async def get_collection_info(self, collection_name: str):
        """Get collection information."""
        async with self._ensure_connection():
            if await self.client.collection_exists(collection_name):
                return await self.client.get_collection(collection_name)
            else:
                logger.warning(f"Collection '{collection_name}' does not exist")
                return None

    async def insert_one(self, collection_name: str, vector: List[float], 
                   text: str, metadata: Dict[str, Any] = None, record_id: str = None):
        """Insert a single vector with metadata."""
        
        if record_id is None:
            record_id = str(uuid.uuid4())
            
        payload = {"text": text}
        if metadata:
            payload.update(metadata)

        async with self._ensure_connection():
            if await self.client.collection_exists(collection_name):
                try:
                    await self.client.upsert(
                        collection_name=collection_name,
                        points=[models.PointStruct(
                            id=record_id,
                            vector=vector,
                            payload=payload
                        )]
                    )
                    logger.debug(f"Inserted document '{record_id}' into '{collection_name}'")
                    return record_id
                    
                except Exception as e:
                    logger.error(f"Error inserting into '{collection_name}': {e}")
                    raise
            else:
                raise ValueError(f"Collection '{collection_name}' does not exist")

    async def insert_many(self, collection_name: str, vectors: List[List[float]], 
                    texts: List[str], metadatas: List[Dict[str, Any]] = None, batch_size: int = 100):
        """Insert multiple vectors efficiently with batching."""

        if not await self.is_collection_exist(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        record_ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
        if metadatas is None:
            metadatas = [{}] * len(texts)

        total_inserted = 0

        async with self._ensure_connection():
            try:
                # Process in batches for better performance
                for i in range(0, len(texts), batch_size):
                    batch_end = min(i + batch_size, len(texts))
                    batch_points = []
                    
                    for j in range(i, batch_end):
                        payload = {"text": texts[j]}
                        if metadatas[j]:
                            payload.update(metadatas[j])
                            
                        point = models.PointStruct(
                            id=record_ids[j],
                            vector=vectors[j],
                            payload=payload
                        )
                        batch_points.append(point)

                    await self.client.upsert(
                        collection_name=collection_name,
                        points=batch_points,
                    )
                    
                    batch_size_actual = len(batch_points)
                    total_inserted += batch_size_actual

                    logger.info(f"Inserted batch {i//batch_size + 1}: {batch_size_actual} documents into '{collection_name}'")

                logger.info(f"Successfully inserted {total_inserted} documents into '{collection_name}'")
                return record_ids[:total_inserted]

            except Exception as e:
                logger.error(f"Error batch inserting into '{collection_name}': {e}")
                raise

    async def search(self, collection_name: str, query_vector: List[float], 
               top_k: int = 5, filter_conditions: Dict[str, Any] = None):
        """Search for similar vectors with optional filtering."""
        
        async with self._ensure_connection():
            if await self.client.collection_exists(collection_name):
                try:
                    search_filter = None
                    if filter_conditions:
                        search_filter = models.Filter(**filter_conditions)

                    results = await self.client.search(
                        collection_name=collection_name,
                        query_vector=query_vector,
                        limit=top_k,
                        query_filter=search_filter,
                        with_payload=True,
                        with_vectors=False  # Don't return vectors to save bandwidth
                    )

                    logger.debug(f"Search completed in '{collection_name}', found {len(results)} results")
                    return results

                except Exception as e:
                    logger.error(f"Error searching in '{collection_name}': {e}")
                    raise
            else:
                logger.warning(f"Collection '{collection_name}' does not exist")
                raise ValueError(f"Collection '{collection_name}' does not exist")

    async def delete_collection(self, collection_name: str):
        """Delete a collection."""
        try:
            async with self._ensure_connection():
                if await self.client.collection_exists(collection_name):
                    await self.client.delete_collection(collection_name)

                    msg= f"Collection '{collection_name}' deleted"
                    logger.info(msg)
                    return {"success": True, "message": msg}
                else:
                    msg = f"Collection '{collection_name}' does not exist"
                    logger.warning(msg)
                    return {"success": False, "message": msg}
                
        except Exception as e:
            msg = f"Error deleting collection '{collection_name}': {e}"
            return {"success": False, "message": msg}

    async def delete_asset_chunks(self, collection_name: str, asset_id: str) -> Dict[str, Any]:
        """Delete all chunks (points) belonging to a specific asset from a collection."""
        try:
            async with self._ensure_connection():
                if not await self.client.collection_exists(collection_name):
                    msg = f"Collection '{collection_name}' does not exist"
                    logger.warning(msg)
                    return {"success": False, "message": msg}

                await self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="asset_id",
                                    match=models.MatchValue(value=asset_id)
                                )
                            ]
                        )
                    )
                )

                msg = f"Deleted points from '{collection_name}' with asset_id: {asset_id}"
                logger.info(msg)
                return {"success": True, "message": msg}

        except Exception as e:
            msg = f"Error deleting points from '{collection_name}' with asset_id={asset_id}: {e}"
            logger.error(msg)
            return {"success": False, "message": msg}