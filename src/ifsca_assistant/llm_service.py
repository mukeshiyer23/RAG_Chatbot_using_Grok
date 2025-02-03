import os
from groq import AsyncGroq
from typing import List, Dict, Optional


class LLMService:
    """Service for handling LLM interactions using Groq"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LLM service with Groq"""
        self.client = AsyncGroq(
            api_key=api_key or os.getenv("GROQ_API_KEY"),
        )
        self.model = "mixtral-8x7b-32768"  # Using Mixtral for better performance

    async def generate_response(self, query: str, context: List[Dict]) -> str:
        """Generate a response using the provided context"""
        context_text = "\n".join([
            f"Source: {chunk['filename']}\n{chunk['text']}"
            for chunk in context[:3]
        ])

        prompt = f"""You are a regulatory expert assistant for IFSCA (International Financial Services Centres Authority). 
        Use the following context to answer the query. If unsure, state you don't know.

        Context:
        {context_text}

        Query: {query}

        Provide a concise, accurate response citing relevant regulations when possible."""

        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful regulatory expert."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.2,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")