from fastapi import HTTPException
from .BaseController import BaseController
from .DataController import DataController
from .LLMController import LLMController
from langchain_text_splitters import RecursiveCharacterTextSplitter
from routes.schema import *
from stores.LLM import DocumentTypeEnum

import logging
logger = logging.getLogger(__name__)

class VDBController(BaseController):
    def __init__(self, vdb_provider=None,
                 embedding_provider=None,
                 generate_provider=None,
                 summarize_provider=None,
                 template_parser=None):
        
        super().__init__()
        self.project_path = self.get_project_path()
        self.vdb_provider = vdb_provider
        self.data_controller = DataController()
        self.llm_controller = LLMController(
            embedding_provider=embedding_provider,
            generate_provider=generate_provider,
            summarize_provider=summarize_provider,
            template_parser=template_parser
        )

    async def get_vdb_health(self) -> HealthResponse:
        health_status = await self.vdb_provider.health_check()
        if health_status:
            return HealthResponse(
                initialized=True,
                message=f"{self.app_settings.VECTOR_DB_HOST}:{self.app_settings.VECTOR_DB_PORT}"
            )
        return HealthResponse(
            initialized=False,
            message="Vector DB not initialized"
        )

    async def get_chunks(self, file_id: str, file_content: list = None, chunk_size: int = None, chunk_overlap: int = None):
        """
        Extract and split file content into chunks.
        Returns a list of document chunks with text and metadata.
        """
        # Use default values if not provided
        chunk_size = self.app_settings.TEXT_CHUNK_SIZE if chunk_size is None else chunk_size
        chunk_overlap = self.app_settings.TEXT_CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap

        if file_content is None:
            logger.info(f"Extracting content for file: {file_id}")
            file_content = await self.data_controller.get_file_content(file_id)
        
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap, 
                length_function=len
            )

            texts = [doc.page_content for doc in file_content]
            metadatas = [doc.metadata for doc in file_content]

            chunks = splitter.create_documents(texts, metadatas=metadatas)
            
            logger.info(f"Created {len(chunks)} chunks from file {file_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks from file {file_id}: {e}")
            return []

    async def process_and_store_chunks(self, file_id: str, asset_id: str,
                                       file_content: list = None,
                                       chunk_size: int = None, chunk_overlap: int = None,
                                       batch_size: int = 50):

        try:
            # Extract and chunk content
            logger.info(f"Processing file {file_id} for asset {asset_id}")
            chunks = await self.get_chunks(file_id, file_content, chunk_size, chunk_overlap)
            
            if not chunks:
                return {
                    "success": False,
                    "message": "No chunks created from file",
                    "chunk_count": 0,
                    "embeddings_count": 0,
                    "inserted_count": 0
                }

            # Extract text from LangChain Document objects
            texts = [chunk.page_content for chunk in chunks]

            # Get embeddings in batches to avoid memory issues
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                try:
                    batch_embeddings = await self.llm_controller.embed_text_batch(texts=batch_texts, document_type=DocumentTypeEnum.DOCUMENT.value)
                    all_embeddings.extend(batch_embeddings)
                    logger.info(f"Generated embeddings for batch {i//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                    raise   # Fail fast

            print(f"Total embeddings generated: {len(all_embeddings)}")
            print("----------")
            print(f"Total texts processed: {len(texts)}")

            if not all_embeddings:
                logger.error(f"No embeddings generated for file {file_id}")
                return {
                    "success": False,
                    "message": "No embeddings generated",
                    "chunk_count": len(chunks),
                    "embeddings_count": 0,
                    "inserted_count": 0
                }
            if len(all_embeddings) != len(texts):
                return {
                    "success": False,
                    "message": "Mismatch between embeddings and texts",
                    "chunk_count": len(chunks),
                    "embeddings_count": len(all_embeddings),
                    "inserted_count": 0
                }

            collection_name = self.app_settings.VECTOR_DB_COLLECTION
            
            # Ensure collection exists
            if not await self.vdb_provider.is_collection_exist(collection_name):
                embedding_size = self.app_settings.EMBEDDING_SIZE
                await self.vdb_provider.create_collection(collection_name, embedding_size)

            # Prepare metadata for each chunk
            metadatas = []
            for i, chunk in enumerate(chunks):                
                metadata = {
                    "chunk_index": i,
                    "file_id": file_id,
                    "asset_id": asset_id,
                }
                
                # Add original metadata from document
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    metadata.update(chunk.metadata)
                    
                metadatas.append(metadata)
            
            # Store in vector DB
            record_ids = await self.vdb_provider.insert_many(
                collection_name=collection_name,
                vectors=all_embeddings,
                texts=texts,
                metadatas=metadatas,
                batch_size=batch_size
            )
            
            inserted_count = len(record_ids) if record_ids else 0
            success = inserted_count > 0

            return {
                "success": success,
                "message": "File processed and stored successfully" if success else "Failed to store chunks",
                "chunk_count": len(chunks),
                "embeddings_count": len(all_embeddings),
                "inserted_count": inserted_count
            }

        except Exception as e:
            logger.error(f"Error in process_and_store_chunks: {e}")
            return {
                "success": False,
                "message": str(e),
                "chunk_count": 0,
                "embeddings_count": 0,
                "inserted_count": 0
            }

    async def search_chunks(self, query: str, top_k: int = 10, similarity_threshold: float = 0.7):
        """
        Search for similar chunks using vector similarity.
        Returns structured search results with metadata.
        """
        try:
            collection_name = self.app_settings.VECTOR_DB_COLLECTION

            # Generate query embedding
            query_vector = await self.llm_controller.embed_text(text=query, document_type=DocumentTypeEnum.QUERY.value)

            # Search in vector DB
            results = await self.vdb_provider.search(
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=top_k,
            )
            if not results:
                logger.info(f"No similar chunks found for query '{query}'")
                return {
                    "success": True,
                    "message": "No similar chunks found",
                    "results": [],
                    "count": 0
                }
            
            # Filter results by similarity threshold
            filtered_results = [
                {
                    "id": result.id,
                    "text": result.payload.get("text", ""),
                    "score": result.score,
                    "metadata": result.payload,
                }
                for result in results
                if result.score >= similarity_threshold
            ]

            logger.info(f"Found {len(filtered_results)} similar chunks for query '{query}'")

            return {
                "success": True,
                "message": "Successfully retrieved similar chunks",
                "results": filtered_results,
                "count": len(filtered_results)
            }
        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            return {
                "success": False,
                "message": str(e),
                "results": [],
                "count": 0
            }

    async def delete_asset_chunks(self, collection_name: str, asset_id: str) -> DeleteAssetResponse:

        result = await self.vdb_provider.delete_asset_chunks(collection_name, asset_id)
        return DeleteAssetResponse(
            success= result['success'],
            message= result['message'],
            asset_id= asset_id
        )

    async def delete_collection(self, collection_name: str) -> DeleteCollectionResponse:

        result = await self.vdb_provider.delete_collection(collection_name)
        return DeleteCollectionResponse(
            success= result['success'],
            message= result['message'],
            collection_name= collection_name
        )

    async def get_all_collections(self) -> CollectionsResponse:
        """
        List all collections in the vector database.
        """
        try: 
            collections = await self.vdb_provider.get_all_collections()
            return CollectionsResponse(
                success=True,
                message="Successfully retrieved all collections",
                collections=collections,
                count=len(collections)
            )
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return CollectionsResponse(
                success=False,
                message=str(e),
                collections=[],
                count=0
            )

    async def get_collection_info(self, collection_name: str) -> CollectionInfoResponse:
        """
        Get detailed info about a specific collection.
        """
        try:
            info = await self.vdb_provider.get_collection_info(collection_name)
            if info:
                return CollectionInfoResponse(
                    success=True,
                    message="Successfully retrieved collection info",
                    collection_name=collection_name,
                    vector_count=info.vectors_count,
                    indexed_vector_count=info.indexed_vectors_count,
                    points_count=info.points_count,
                    payload_schema=info.payload_schema
                )

            return CollectionInfoResponse(
                success=False,
                message=f"Collection '{collection_name}' does not exist",
                collection_name=collection_name,
                vector_count=None,
                indexed_vector_count=None,
                points_count=None,
                payload_schema=None
            )
            
        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {e}")
            return CollectionInfoResponse(
                success=False,
                message=str(e),
                collection_name=collection_name,
                vector_count=None,
                indexed_vector_count=None,
                points_count=None,
                payload_schema=None
            )