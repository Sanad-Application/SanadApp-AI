from fastapi import APIRouter, Depends, UploadFile, status, HTTPException, Request
from fastapi.responses import JSONResponse
import aiofiles
from helpers.config import get_settings, settings
from controllers import DataController, LLMController, VDBController
from .schema import *
import uuid

import logging
logger = logging.getLogger(__name__)

summary_router = APIRouter(prefix="/summary", tags=["Summary"])

@summary_router.post("/text", response_model=SummaryResponse)
async def summarize_text(
    summary_request: SummarizeTextRequest,
    request: Request
):
    
    try:
        llm_controller = LLMController(
            embedding_provider=request.app.embedding_client,
            generate_provider=request.app.generation_client,
            summarize_provider=request.app.summarization_client,
            template_parser=request.app.template_parser
        )
        
        # Generate summary
        summary = await llm_controller.summarize_text(summary_request.text)
        logger.info(f"Text summarized successfully.")

        return SummaryResponse(
            success=True,
            message="Text summarized successfully",
            summary=summary,
        )
        
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        raise HTTPException(status_code=500, detail=f"Error summarizing text: {str(e)}")

@summary_router.post("/file/", response_model=SummaryResponse)
async def summarize_uploaded_file(
    file: UploadFile,
    request: Request,
    app_settings: settings = Depends(get_settings)
):
    
    try:
        # Initialize controllers
        data_controller = DataController()
        vdb_controller = VDBController(
            vdb_provider=request.app.vdb_client,
            embedding_provider=request.app.embedding_client,
            generate_provider=request.app.generation_client,
            summarize_provider=request.app.summarization_client,
            template_parser=request.app.template_parser
        )        
        llm_controller = LLMController(
            embedding_provider=request.app.embedding_client,
            generate_provider=request.app.generation_client,
            summarize_provider=request.app.summarization_client,
            template_parser=request.app.template_parser
        )
        
        # Validate file
        is_valid, message = data_controller.validfile(file=file)
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST, 
                content={"message": message}
            )
        
        # Save file temporarily
        file_path, file_id = data_controller.get_file_path(filename=file.filename)
        
        async with aiofiles.open(file_path, "wb") as f:
            while True:
                chunk = await file.read(app_settings.PDF_CHUNK_SIZE)
                if not chunk:
                    break
                await f.write(chunk)
        
        # Extract text content
        file_content = await data_controller.get_file_content(file_id)
        if not file_content:
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        full_text = "\n\n".join([doc.page_content for doc in file_content])

        # Store content in vector DB
        asset_id = str(uuid.uuid4())
        _ = await vdb_controller.process_and_store_chunks(
            file_id=file_id,
            asset_id=asset_id,
            file_content=file_content,
            chunk_size=app_settings.TEXT_CHUNK_SIZE,
            chunk_overlap=app_settings.TEXT_CHUNK_OVERLAP
        )
    
        # Generate summary
        summary = await llm_controller.summarize_text(full_text)
        
        logger.info(f"File summarized successfully: {file.filename}")
        return SummaryResponse(
            success=True,
            message="File summarized successfully",
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error summarizing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error summarizing file: {str(e)}")