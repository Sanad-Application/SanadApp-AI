from ..LLMInterface import LLMInterface
from ..LLMEnums import GeminiEnums, DocumentTypeEnum
import google.generativeai as genai
import logging


class GeminiProvider(LLMInterface):
    def __init__(self, api_key: str,
                 default_max_input_characters: int=1000,
                 default_max_output_tokens: int=1000,
                 temperature: float=0.7):
        
        self.api_key = api_key
        self.default_max_input_characters = default_max_input_characters
        self.default_max_output_tokens = default_max_output_tokens
        self.temperature = temperature
        
        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.enums = GeminiEnums
        
        genai.configure(api_key=self.api_key)
        
        self.logger = logging.getLogger(__name__)
    
    async def set_generation_model(self, generation_model_id: str):
        self.generation_model_id = generation_model_id
    
    async def set_embedding_model(self, embedding_model_id: str, embedding_size: int):
        self.embedding_model_id = embedding_model_id
        self.embedding_size = embedding_size
    
    async def process_text(self, text: str):
        return text[:self.default_max_input_characters].strip()
    
    async def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None, temperature: float=None):
        if not self.generation_model_id:
            self.logger.error("Generation model for Gemini was not set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_max_output_tokens
        temperature = temperature if temperature else self.temperature
        
        try:
            model = genai.GenerativeModel(model_name=self.generation_model_id)
            
            chat = model.start_chat(history=chat_history)
            response = chat.send_message(
                await self.process_text(prompt),
                generation_config={
                    'max_output_tokens': max_output_tokens,
                    'temperature': temperature})
            
            if not response or not response.text:
                self.logger.error("Error while generating text with Gemini")
                return None
            
            return response.text
        
        except Exception as e:
            self.logger.error(f"Error generating text with Gemini: {str(e)}")
            return None

    async def embed_text(self, text: str, document_type: str = None):
        if not self.embedding_model_id:
            self.logger.error("Embedding model for Gemini was not set")
            return None
        
        try:
            task_type= self.enums.DOCUMENT.value
            if document_type == DocumentTypeEnum.QUERY.value:
                task_type= self.enums.QUERY.value

            embeddings= genai.embed_content(
                model= self.embedding_model_id,
                content= await self.process_text(text),
                task_type= task_type)
            
            if not embeddings or not embeddings.embedding:
                self.logger.error("Error while embedding text with Gemini")
                return None
            
            return embeddings.embedding
        
        except Exception as e:
            self.logger.error(f"Error embedding text with Gemini: {str(e)}")
            return None
    
    async def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "parts": [await self.process_text(prompt)]
        }