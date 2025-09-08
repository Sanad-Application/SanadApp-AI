from fastapi import FastAPI
import uvicorn
from routes import base, chat, data, summary
from stores.LLM import LLMFactory
from stores.VectorDB import VDBFactory
from stores.LLM.templates import TemplateParser
from helpers.config import get_settings
settings = get_settings()

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LegalBot API",
    description="Legal document processing and retrieval system with vector search",
    version="0.1.0"
)

@app.on_event("startup")
async def startup_db():
    global vdb_instance
    try:
        vdb_factory = VDBFactory()
        app.vdb_client = vdb_factory.create(settings.VECTOR_DB_BACKEND)
        await app.vdb_client.connect()

        if await app.vdb_client.health_check():
            logger.info("✅ Vector DB connection established and healthy")
        else:
            logger.warning("⚠️ Vector DB connection established but health check failed")

        # generation client
        llm_provider_factory = LLMFactory()
        app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
        await app.generation_client.set_generation_model(generation_model_id=settings.GENERATION_MODEL_ID)

        # embedding client
        app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
        await app.embedding_client.set_embedding_model(embedding_model_id=settings.EMBEDDING_MODEL_ID,
                                                       embedding_size=settings.EMBEDDING_SIZE)

        # summarization client
        app.summarization_client = llm_provider_factory.create(provider=settings.SUMMARIZATION_BACKEND)
        await app.summarization_client.set_summarization_model(summarization_model_id=settings.SUMMARIZATION_MODEL_ID)

        # template parser
        app.template_parser = TemplateParser(lang=settings.PRIMARY_LANGUAGE,
                                            default_lang=settings.DEFAULT_LANGUAGE)
        
        logger.info("Application startup completed")

    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise
        
@app.on_event("shutdown")
async def shutdown_db():
    try:
        await app.vdb_client.disconnect()
        logger.info("🛑 Vector DB connection closed successfully")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")
        raise

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "SanadApp API",
        "version": "0.1.0"
    }

# Include routers
app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(chat.chat_router)
app.include_router(summary.summary_router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
