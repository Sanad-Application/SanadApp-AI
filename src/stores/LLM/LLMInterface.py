from abc import ABC, abstractmethod

class LLMInterface(ABC):

    @abstractmethod
    async def set_generation_model(self, generation_model_id: str):
        pass

    @abstractmethod
    async def set_embedding_model(self, embedding_model_id: str, embedding_size: int):
        pass

    @abstractmethod
    async def process_text(self, text: str):
        pass

    @abstractmethod
    async def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None, temperature: float = None):
        pass

    @abstractmethod
    async def embed_text(self, text: str):
        pass

    @abstractmethod
    async def construct_prompt(self, prompt: str, role: str):
        pass
