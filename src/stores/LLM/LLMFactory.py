from .LLMEnums import OpenAIEnums, CoHereEnums, GeminiEnums, LLMModel, DocumentTypeEnum
from .providers import OpenAIProvider, CoHereProvider, GeminiProvider

class LLMFactory:

    @classmethod
    def create(cls, provider: str):

        from helpers.config import get_settings
        settings = get_settings()
        
        if provider == LLMModel.OPENAI.value:
            return OpenAIProvider(
                api_key = settings.OPENAI_API_KEY,
                api_url = settings.OPENAI_API_URL,
                default_max_input_characters=settings.DEFAULT_MAX_INPUT_CHARACTERS,
                default_max_output_tokens=settings.DEFAULT_MAX_OUTPUT_TOKENS,
                temperature=settings.DEFAULT_TEMPERATURE
            )

        if provider == LLMModel.COHERE.value:
            return CoHereProvider(
                api_key = settings.COHERE_API_KEY,
                default_max_input_characters=settings.DEFAULT_MAX_INPUT_CHARACTERS,
                default_max_output_tokens=settings.DEFAULT_MAX_OUTPUT_TOKENS,
                temperature=settings.DEFAULT_TEMPERATURE
            )

        if provider == LLMModel.GEMINI.value:
            return GeminiProvider(
                api_key = settings.GEMINI_API_KEY,
                default_max_input_characters=settings.DEFAULT_MAX_INPUT_CHARACTERS,
                default_max_output_tokens=settings.DEFAULT_MAX_OUTPUT_TOKENS,
                temperature=settings.DEFAULT_TEMPERATURE
            )

        return None