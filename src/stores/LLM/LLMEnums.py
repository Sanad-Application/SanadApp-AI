from enum import Enum

class LLMModel(Enum):
    OPENAI="openai"
    COHERE="cohere"
    GEMINI="gemini"

class OpenAIEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class CoHereEnums(Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "CHATBOT"

    DOCUMENT = "search_document"
    QUERY = "search_query"

class GeminiEnums(Enum):
    USER = "user"
    ASSISTANT = "model"

    DOCUMENT = "retrieval_document"
    QUERY = "retrieval_query"


class DocumentTypeEnum(Enum):
    DOCUMENT = "document"
    QUERY = "query"

