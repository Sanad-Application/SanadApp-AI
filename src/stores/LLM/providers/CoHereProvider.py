from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums, DocumentTypeEnum
import cohere
import logging

class CoHereProvider(LLMInterface):

    def __init__(self, api_key: str,
                default_max_input_characters: int=1000,
                default_max_output_tokens: int=1000,
                default_temperature: float=0.1):
        
        self.api_key = api_key
        self.default_max_input_characters = default_max_input_characters
        self.default_max_output_tokens = default_max_output_tokens
        self.default_temperature = default_temperature

        self.generation_model_id = None
        self.summarization_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.enums = CoHereEnums

        self.client = cohere.Client(api_key=self.api_key)

        self.logger = logging.getLogger(__name__)

    async def set_generation_model(self, generation_model_id: str):
        self.generation_model_id = generation_model_id

    async def set_summarization_model(self, summarization_model_id: str):
        self.summarization_model_id = summarization_model_id

    async def set_embedding_model(self, embedding_model_id: str, embedding_size: int):
        self.embedding_model_id = embedding_model_id
        self.embedding_size = embedding_size

    async def process_text(self, text: str):
        return text[:self.default_max_input_characters].strip()

    async def _chat_completion(self, user_prompt: str, system_prompt: str, model_id: str,
                               temperature: float = None, max_output_tokens: int = None,
                               ):
        """Common method for chat completion used by both generate_text and summarize_text"""
        if not self.client:
            self.logger.error("CoHere client was not set")
            return None

        if not model_id:
            self.logger.error("No model provided for chat completion")
            return None
        
        try:
            chat_history = [await self.construct_prompt(
                prompt=system_prompt,
                role=self.enums.SYSTEM.value
            )]
            response = self.client.chat(
                model=model_id,
                chat_history=chat_history,
                message=await self.process_text(user_prompt),
                temperature=temperature or self.default_temperature,
                max_tokens=max_output_tokens or self.default_max_output_tokens,
            )

            if not response or not response.text:
                self.logger.error("Error while generating text with CoHere")
                return None
            
            return response.text
        except Exception as e:
            self.logger.error(f"Error in chat completion with CoHere: {str(e)}")
            return None

    async def generate_text(self, user_prompt: str, system_prompt: str):
        if not self.generation_model_id:
            self.logger.error("Generation model for CoHere was not set")
            return None
        
        return await self._chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model_id=self.generation_model_id
        )

    async def embed_text(self, text: str, document_type: str = None):

        if not self.client:
            self.logger.error("CoHere client was not set")
            return None
        
        if not self.embedding_model_id:
            self.logger.error("Embedding model for CoHere was not set")
            return None

        input_type = self.enums.DOCUMENT
        if document_type == DocumentTypeEnum.QUERY:
            input_type = self.enums.QUERY

        response = self.client.embed(
            model = self.embedding_model_id,
            texts = text,
            input_type = input_type,
            embedding_types=['float'],
        )

        if not response or not response.embeddings or not response.embeddings.float:
            self.logger.error("Error while embedding text with CoHere")
            return None
        
        return response.embeddings.float[0]

    async def summarize_text(self, user_prompt: str, system_prompt: str):

        if not self.summarization_model_id:
            self.logger.error("No model set for summarization with CoHere")
            return None

        return await self._chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model_id=self.summarization_model_id,
            temperature=0.3  # Lower temperature for more focused summarization
        )
    
    async def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "text": await self.process_text(prompt)
        }