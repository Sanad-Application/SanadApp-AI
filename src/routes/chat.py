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
        
        vdb_controller = VDBController(request.app.vdb_client)
        result = vdb_controller.get_vdb_health()

        status_code = 200 if result.initialized else 500
        return JSONResponse(content=result.dict(), status_code=status_code)

@chat_router.post("/answer", response_model=AnswerResponse)
async def generate_answer(request: Request, answer_request: AnswerRequest, app_settings: settings = Depends(get_settings)):
    """
    Generate text based on the provided prompt and chat history.
    """
    try:
        vdb_controller = VDBController(request.app.vdb_client)
        chunks_result = await vdb_controller.search_chunks(
            query=answer_request.query,
            top_k=10,
            similarity_threshold=0.7
            )

        llm_controller = LLMController(request.app.embedding_client,
                                       request.app.generation_client,
                                       request.app.template_parser)
        answer, full_prompt, chat_history = await llm_controller.generate_text(answer_request.query, chunks_result)

        if not answer:
            return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "signal": "RAG answer error"
                    }
            )
    
        return JSONResponse(
            content={
                "signal": "RAG answer success",
                "answer": answer,
                "full_prompt": full_prompt,
                "chat_history": chat_history
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))