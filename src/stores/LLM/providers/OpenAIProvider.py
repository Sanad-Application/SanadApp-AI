from openai import OpenAI
from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums
import logging


class OpenAIProvider(LLMInterface):
    def __init__(self,
                api_key: str,
                api_url: str = None,
                default_max_input_characters: int=1000,
                default_max_output_tokens: int = 1000, 
                default_temperature: float = 0.5):
        
        self.api_key = api_key
        self.api_url = api_url
        self.default_max_output_tokens = default_max_output_tokens
        self.default_max_input_characters = default_max_input_characters
        self.default_temperature = default_temperature

        self.generation_model_id = None
        self.summarization_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.enums = OpenAIEnums

        self.client = OpenAI(api_key=api_key, api_base=api_url)

        self.logger = logging.getLogger(__name__)


    async def set_generation_model(self, generation_model_id: str):
        self.generation_model_id = generation_model_id

    async def set_summarization_model(self, model_id: str):
        self.summarization_model_id = model_id

    async def set_embedding_model(self, embedding_model_id: str, embedding_size: int):
        self.embedding_model_id = embedding_model_id
        self.embedding_size = embedding_size

    async def process_text(self, text: str):
        return text[:self.default_max_input_characters].strip()

    async def _chat_completion(self, prompt: str, chat_history: list = [], 
                               temperature: float = None, max_output_tokens:int = None,
                               model_id: str = None):
        """Common method for chat completion used by both generate_text and summarize_text"""
        if self.client is None:
            self.logger.error("OpenAI client is not initialized.")
            return None

        if not model_id:
            self.logger.error("No model provided for chat completion")
            return None

        try:
            # Make a copy of chat_history to avoid modifying the original
            messages = chat_history.copy()
            messages.append(await self.construct_prompt(prompt, self.enums.USER.value))

            response = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature or self.default_temperature
            )

            if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
                self.logger.error("Error while generating text with OpenAI")
                return None
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error in chat completion with OpenAI: {str(e)}")
            return None

    async def generate_text(self, prompt: str, chat_history: list = []):
        if self.generation_model_id is None:
            self.logger.error("Generation model ID is not set.")
            return None

        return await self._chat_completion(
            prompt=prompt,
            chat_history=chat_history,
            model_id=self.generation_model_id
        )

    async def embed_text(self, text: str, embedding_size: int = None):
        if self.embedding_model_id is None:
            self.logger.error("Embedding model ID is not set.")
            return None

        if self.client is None:
            self.logger.error("OpenAI client is not initialized.")
            return None
        
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model_id,
            )
        
        if not response or not response.data or len(response.data) == 0 or not response.data[0].embedding:
            self.logger.error("Error while embedding text with OpenAI")
            return None
        
        return response.data[0].embedding
    
    async def summarize_text(self, user_prompt: str, system_prompt: str = "", chat_history: list = []):

        if self.summarization_model_id is None:
            self.logger.error("Summary model ID is not set.")
            return None

        return await self._chat_completion(
            prompt=user_prompt,
            chat_history=chat_history,
            temperature=0.3,  # Lower temperature for more focused summarization
            model_id=self.summarization_model_id
        )

    async def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": await self.process_text(prompt)
        }