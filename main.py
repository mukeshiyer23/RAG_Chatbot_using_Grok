import os
import asyncio
from typing import List, Dict, Any
import logging
import redis

from groq import Groq
import spacy

from src.document_parser import DocumentParser
from src.embedding import EmbeddingProcessor


class IFSCARegulatoryChatbot:
    def __init__(self,
                 groq_api_key: str,
                 redis_url: str = 'redis://localhost:6379'):
        """Initialize chatbot with advanced configurations"""
        # LLM Configuration
        self.groq_client = Groq(api_key=groq_api_key)
        self.model = "llama3-8b-8192"

        # Document Processing
        self.embedding_processor = EmbeddingProcessor()
        self.document_parser = DocumentParser(self.embedding_processor)

        # Caching Layer
        self.redis_client = redis.Redis.from_url(redis_url)

        # NLP Processing
        spacy.cli.download('en_core_web_sm')
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.max_length = 2000000

    async def initialize_document_embeddings(self):
        """
        Process and embed documents from data folder
        """
        try:
            # Parse documents
            await self.document_parser.parse_documents()
            print("Document embeddings processed successfully.")
        except Exception as e:
            logging.error(f"Document initialization error: {e}")

    async def chat(self, query: str) -> Dict[str, Any]:
        """Advanced semantic search and response generation"""
        try:
            # Semantic search
            relevant_chunks = await self.embedding_processor.semantic_search(query)

            if not relevant_chunks:
                return await self._handle_ambiguous_query(query)

            # Generate response using LLM
            response = await self._generate_response(query, relevant_chunks)

            return {
                "status": "success",
                "answer": response,
                "sources": [chunk['filename'] for chunk in relevant_chunks]
            }

        except Exception as e:
            logging.error(f"Chat processing error: {e}")
            return {
                "status": "error",
                "answer": "Unable to process query.",
                "sources": []
            }

    async def _generate_response(self, query: str, chunks: List[Dict], model: str = "llama3-8b-8192"):
        """Generate response from semantic search results"""
        truncated_chunks = chunks[:3]  # Limit to top 3 chunks

        context = "\n\n".join([f"Source: {chunk['filename']}\nText: {chunk['text']}" for chunk in truncated_chunks])

        prompt = f"""Concisely answer the query using the provided context.

        Context:
        {context}

        Query: {query}

        Provide a brief, direct response."""

        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model=model,
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Response generation error: {e}")
            return "Unable to generate a response."

    async def _handle_ambiguous_query(self, query: str) -> Dict[str, Any]:
        """Handle unclear queries with clarification"""
        clarification_prompt = f"""
        Analyze the following query and generate clarifying questions:
        Query: {query}

        Provide:
        1. Possible interpretations
        2. Clarifying questions
        3. Refinement strategies
        """

        response = self.groq_client.chat.completions.create(
            messages=[{"role": "system", "content": clarification_prompt}],
            model=self.model,
            temperature=0.3,
            max_tokens=300
        )

        return {
            "status": "clarification_needed",
            "response": response.choices[0].message.content
        }


async def main():
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        logging.error("GROQ_API_KEY not set")
        return

    bot = IFSCARegulatoryChatbot(groq_api_key)

    # Initialize document embeddings
    await bot.initialize_document_embeddings()

    while True:
        try:
            query = input("\nYour query (or 'exit'): ").strip()
            if query.lower() in ['exit', 'quit', 'bye']:
                break

            response = await bot.chat(query)
            print(f"\nStatus: {response.get('status', 'Unknown')}")
            print(f"Response: {response.get('answer', 'No response')}")

            if response.get('sources'):
                print("\nSources:")
                for source in response['sources']:
                    print(f"- {source}")

        except Exception as e:
            logging.error(f"Main loop error: {e}")
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())