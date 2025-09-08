from pydantic_settings import BaseSettings, SettingsConfigDict

class settings(BaseSettings):

    # Application settings
    APP_NAME: str = "SanadApp"
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

    OPENAI_API_KEY: str = ""
    OPENAI_API_URL: str = ""
    COHERE_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    GENERATION_MODEL_ID: str = ""
    SUMMARIZATION_MODEL_ID: str = ""
    EMBEDDING_MODEL_ID: str = ""
    EMBEDDING_SIZE: int = 1024
    DEFAULT_MAX_INPUT_CHARACTERS: int = 1024
    DEFAULT_MAX_OUTPUT_TOKENS: int = 200
    DEFAULT_TEMPERATURE: float = 0.1

    # Template settings
    DEFAULT_LANGUAGE: str = "ar"
    PRIMARY_LANGUAGE: str = "ar"

    # Vector DB settings
    VECTOR_DB_BACKEND: str
    VECTOR_DB_HOST: str = "localhost"
    VECTOR_DB_PORT: int = 6333
    VECTOR_DB_GRPC_PORT: int = 6334
    VECTOR_DB_DISTANCE_METHOD: str
    VECTOR_DB_COLLECTION: str = "sanadapp"

    class Config:
        env_file = "src/.env"

def get_settings():
    return settings()
