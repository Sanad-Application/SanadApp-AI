from .BaseController import BaseController
from typing import List, Tuple, Dict
import asyncio

import logging
logger = logging.getLogger(__name__)

class LLMController(BaseController):
    def __init__(self, embedding_provider=None, generate_provider=None, template_parser=None):
        super().__init__()
        self.embedding_provider = embedding_provider 
        self.generate_provider = generate_provider
        self.template_parser = template_parser

    def get_embedding_sync(self, text: str) -> List[float]:
        """Get embedding for text (synchronous wrapper)."""
        try:
            return self.embedding_provider.embed_text(text)
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    async def get_embedding_async(self, text: str) -> List[float]:
        """Get embedding for text asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, 
            self.get_embedding_sync, 
            text
        )
    
    async def get_embeddings_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts efficiently."""
        try:
            loop = asyncio.get_event_loop()
            # Process each text individually since embed_text takes single text
            embeddings = []
            for text in texts:
                embedding = await loop.run_in_executor(None,
                    self.embedding_provider.embed_text,
                    text
                )
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Error getting batch embeddings: {e}")
            raise

    async def generate_text(self, query: str, chunks_result: List[Dict]) -> Tuple[str, str, List]:
        """Generate text based on query and chat history."""
        try:
            system_prompt = self.template_parser.get("rag", "system_prompt")

            documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                    "doc_num": idx + 1,
                    "chunk_text": doc.text,
                })
                for idx, doc in enumerate(chunks_result)
            ])

            footer_prompt = self.template_parser.get("rag", "footer_prompt", {"query": query})

            chat_history = [
                self.generate_provider.construct_prompt(
                    prompt=system_prompt,
                    role=self.generate_provider.enums.SYSTEM.value,
                )
            ]

            full_prompt = "\n\n".join([documents_prompts, footer_prompt])

            # step4: Retrieve the Answer
            answer = self.generate_provider.generate_text(
                prompt=full_prompt,
                chat_history=chat_history
            )

            return answer, full_prompt, chat_history

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise