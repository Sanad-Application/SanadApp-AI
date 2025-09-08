from fastapi import APIRouter, HTTPException, Depends,status, Request
from fastapi.responses import JSONResponse
from controllers import VDBController, LLMController
from helpers.config import get_settings, settings
from .schema import *
import logging
logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/chat", tags=["Chat"])

@chat_router.get("/health", response_model= HealthResponse)
async def get_vdb_health(request: Request):
        
        vdb_controller = VDBController(
            vdb_provider=request.app.vdb_client,
            embedding_provider=request.app.embedding_client,
            generate_provider=request.app.generation_client,
            summarize_provider=request.app.summarization_client,
            template_parser=request.app.template_parser
        )
        result = await vdb_controller.get_vdb_health()

        status_code = 200 if result.initialized else 500
        return JSONResponse(content=result.dict(), status_code=status_code)

@chat_router.post("/", response_model=ChatResponse)
async def generate_answer(request: Request, chat_request: ChatRequest, app_settings: settings = Depends(get_settings)):
    """
    Generate text based on the provided prompt and chat history.
    """
    try:
        vdb_controller = VDBController(
            vdb_provider=request.app.vdb_client,
            embedding_provider=request.app.embedding_client,
            generate_provider=request.app.generation_client,
            summarize_provider=request.app.summarization_client,
            template_parser=request.app.template_parser
        )
        result = await vdb_controller.search_chunks(
            query=chat_request.query,
            top_k=10,
            similarity_threshold=0.7
        )

        if not result.get("success", False):
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "signal": "VDB search error",
                    "message": result.get("message", "Unknown error"),
                    "results": result.get("results", []),
                    "count": result.get("count", 0)
                }
            )

        llm_controller = LLMController(
            embedding_provider=request.app.embedding_client,
            generate_provider=request.app.generation_client,
            summarize_provider=request.app.summarization_client,
            template_parser=request.app.template_parser
        )
        answer = await llm_controller.generate_text(
            query=chat_request.query,
            chunks_result=result.get("results", [])
        )

        if not answer:
            return ChatResponse(
                success=False,
                message="No answer generated from RAG process.",
                answer=None
            )

        return ChatResponse(
            success=True,
            message="RAG answer generated successfully.",
            answer=answer
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))