from pydantic_settings import BaseSettings, SettingsConfigDict

class settings(BaseSettings):

    # Application settings
    APP_NAME: str = "LegalBot"
    APP_VERSION: str

    # File upload settings
    ALLOWED_FILE_TYPES: list
    MAX_FILE_SIZE: int
    PDF_CHUNK_SIZE: int

    # Chunking settings
    TEXT_CHUNK_SIZE: int
    TEXT_CHUNK_OVERLAP: int

    # LLM settings
    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str
    SUMMARIZATION_BACKEND: str

    OPENAI_API_KEY: str = None
    OPENAI_API_URL: str = None
    COHERE_API_KEY: str = None
    GEMINI_API_KEY: str = None

    GENERATION_MODEL_ID: str = None
    SUMMARIZATION_MODEL_ID: str = None
    EMBEDDING_MODEL_ID: str = None
    EMBEDDING_SIZE: int = None
    DEFAULT_MAX_INPUT_CHARACTERS: int = None
    DEFAULT_MAX_OUTPUT_TOKENS: int = None
    DEFAULT_TEMPERATURE: float = None

    # Template settings
    DEFAULT_LANGUAGE: str = "ar"
    PRIMARY_LANGUAGE: str = "ar"

    # Vector DB settings
    VECTOR_DB_BACKEND: str
    VECTOR_DB_HOST: str = "localhost"
    VECTOR_DB_PORT: int = 6333
    VECTOR_DB_GRPC_PORT: int = 6334
    VECTOR_DB_DISTANCE_METHOD: str
    VECTOR_DB_COLLECTION: str = "legalbot"

    class Config:
        env_file = ".env"

def get_settings():
    return settings()
