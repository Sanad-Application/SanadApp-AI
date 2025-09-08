from fastapi import APIRouter, Depends, UploadFile, status, HTTPException, Request
from fastapi.responses import JSONResponse
import aiofiles
from helpers.config import get_settings, settings
from controllers import DataController, VDBController
from .schema import *
import os
import uuid

import logging
logger = logging.getLogger(__name__)

data_router = APIRouter(prefix="/data", tags=["Data"])

@data_router.post("/upload/", response_model=UploadResponse)
async def upload_file(file: UploadFile, request: Request, app_settings: settings = Depends(get_settings)):
    """Upload a file and return file information for processing."""
    
    data_controller = DataController()
    is_valid, message = data_controller.validfile(file=file)
     
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, 
            content={"message": message}
        )
    
    file_path, file_id = data_controller.get_file_path(filename=file.filename)

    try:
        # Save the uploaded file
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.PDF_CHUNK_SIZE):
                await f.write(chunk)
                
        # Generate a unique asset ID
        asset_id = str(uuid.uuid4())

        vdb_controller = VDBController(
            vdb_provider=request.app.vdb_client,
            embedding_provider=request.app.embedding_client,
            generate_provider=request.app.generation_client,
            summarize_provider=request.app.summarization_client,
            template_parser=request.app.template_parser
        )
        result = await vdb_controller.process_and_store_chunks(
            file_id=file_id,
            asset_id=asset_id,
            chunk_size=app_settings.TEXT_CHUNK_SIZE,
            chunk_overlap=app_settings.TEXT_CHUNK_OVERLAP
        )
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result['message'])

        logger.info(f"File uploaded successfully: {file.filename}, asset_id: {asset_id}")
        return UploadResponse(
            success=True,
            message="File uploaded successfully",
            file_name=file.filename,
            asset_id=asset_id,
            file_size=os.path.getsize(file_path),
            chunk_count=result['chunk_count'],
            embeddings_count=result['embeddings_count'],
            inserted_count=result['inserted_count']
        )
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@data_router.delete("collections/{collection_name}/asset/{asset_id}", response_model=DeleteAssetResponse)
async def delete_asset_chunks(request: Request, collection_name: str, asset_id: str):
    """Delete all chunks associated with an asset from collection."""

    vdb_controller = VDBController(
        vdb_provider=request.app.vdb_client,
        embedding_provider=request.app.embedding_client,
        generate_provider=request.app.generation_client,
        summarize_provider=request.app.summarization_client,
        template_parser=request.app.template_parser
    )
    result = await vdb_controller.delete_asset_chunks(collection_name, asset_id)

    if not result.success:
        raise HTTPException(status_code=500, detail=result.message)

    return result

@data_router.delete("/collections/{collection_name}", response_model=DeleteCollectionResponse)
async def delete_collection(request: Request, collection_name: str):
    """Delete all chunks associated with a collection from vector DB."""

    vdb_controller = VDBController(
        vdb_provider=request.app.vdb_client,
        embedding_provider=request.app.embedding_client,
        generate_provider=request.app.generation_client,
        summarize_provider=request.app.summarization_client,
        template_parser=request.app.template_parser
    )
    result = await vdb_controller.delete_collection(collection_name)

    if not result.success:
        raise HTTPException(status_code=500, detail=result.message)

    return result

@data_router.get("/collections", response_model=CollectionsResponse)
async def list_collections(request: Request):
    vdb_controller = VDBController(
        vdb_provider=request.app.vdb_client,
        embedding_provider=request.app.embedding_client,
        generate_provider=request.app.generation_client,
        summarize_provider=request.app.summarization_client,
        template_parser=request.app.template_parser
    )
    result = await vdb_controller.get_all_collections()

    if not result.success:
        raise HTTPException(status_code=500, detail=result.message)

    return result

@data_router.get("/collections/{collection_name}", response_model=CollectionInfoResponse)
async def get_collection_info(request: Request, collection_name: str):

    vdb_controller = VDBController(
        vdb_provider=request.app.vdb_client,
        embedding_provider=request.app.embedding_client,
        generate_provider=request.app.generation_client,
        summarize_provider=request.app.summarization_client,
        template_parser=request.app.template_parser
    )
    result = await vdb_controller.get_collection_info(collection_name)

    if not result.success:
        raise HTTPException(status_code=500, detail=result.message)

    return result
