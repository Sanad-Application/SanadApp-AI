from .BaseController import BaseController
from typing import List, Tuple, Dict
import asyncio

import logging
logger = logging.getLogger(__name__)

class LLMController(BaseController):
    def __init__(self, embedding_provider=None,
                generate_provider=None,
                summarize_provider=None,
                template_parser=None):
        
        super().__init__()
        self.embedding_provider = embedding_provider 
        self.generate_provider = generate_provider
        self.summarize_provider = summarize_provider
        self.template_parser = template_parser


    async def embed_text(self, text: str, document_type: str) -> List[float]:
        """Get embedding for text asynchronously."""
        return await self.embedding_provider.embed_text(text=text, document_type=document_type)

    async def embed_text_batch(self, texts: List[str], document_type: str) -> List[List[float]]:
        """Get embeddings for multiple texts efficiently."""
        try:
            results = await asyncio.gather(
                *[self.embed_text(text, document_type=document_type) for text in texts],
                return_exceptions=True
            )
            return results
        except Exception as e:
            logger.error(f"Error getting batch embeddings: {e}")
            raise

    async def generate_text(self, query: str, chunks_result: List[Dict]) -> str:
        """Generate text based on query and chat history."""
        try:
            system_prompt = self.template_parser.get("rag", "system_prompt")

            documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                    "doc_num": idx + 1,
                    "chunk_text": doc["text"],
                })
                for idx, doc in enumerate(chunks_result)
            ])

            footer_prompt = self.template_parser.get("rag", "footer_prompt", {"query": query})

            user_prompt = "\n\n".join([documents_prompts, footer_prompt])

            # Retrieve the Answer
            answer = await self.generate_provider.generate_text(
                user_prompt=user_prompt,
                system_prompt=system_prompt
            )

            return answer

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    async def summarize_text(self, text: str) -> str:
        """Summarize given text."""
        try:
            print(text[:100])
            system_prompt = self.template_parser.get("summarizer", "system_prompt")
            user_prompt = self.template_parser.get("summarizer", "footer_prompt", {"text": text})

            return await self.summarize_provider.summarize_text(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )

        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            raise