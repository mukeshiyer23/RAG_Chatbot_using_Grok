from typing import List, Dict, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from datetime import datetime
import asyncio
from openai import OpenAI

from src.embedder import Embedder
from src.qdrant_store import QdrantStore
from src.utils.logger import logger
from src.utils.exceptions import DocumentProcessingError

COLLECTION_NAME = "documents"  # Matches collection name from main.py


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, str]]


class ChatServer:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.embedder = Embedder()  # Uses your existing embedder
        self.qdrant_store = QdrantStore()  # Uses your existing QdrantStore
        logger.info("Chat server initialized successfully")

    def _create_system_prompt(self, context: str) -> str:
        return f"""You are an IFSCA (International Financial Services Centres Authority) regulatory assistant. 
Your responses must be based solely on the provided context. If information is not found in the context, 
acknowledge that and suggest contacting IFSCA directly.

Current context from IFSCA documents:
{context}

Guidelines:
1. Only use information from the provided context
2. If information is not in the context, say "Based on the available information, I cannot provide a complete answer to this question. Please contact IFSCA directly for accurate guidance."
3. Always cite the source document when providing information
4. Keep responses clear and structured
5. For regulatory matters, include relevant circular/notification references if available"""

    async def _get_relevant_context(self, query: str, limit: int = 5) -> tuple[str, List[Dict]]:
        """Retrieve relevant context from Qdrant"""
        try:
            # Generate embedding for the query
            query_embedding = self.embedder.embed(query)

            # Search in Qdrant
            results = self.qdrant_store.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=limit
            )

            context = ""
            sources = []

            for result in results:
                # Extract content and metadata
                content = result.metadata.get('content', '')
                file_path = result.metadata.get('file_path', '')
                file_type = result.metadata.get('file_type', '')

                # Add to context
                context += f"\nFrom {file_path}:\n{content}\n"

                # Add to sources if not already included
                source_info = {
                    'file_path': file_path,
                    'file_type': file_type
                }
                if source_info not in sources:
                    sources.append(source_info)

            return context.strip(), sources

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            raise DocumentProcessingError("Failed to retrieve context") from e

    async def generate_response(
            self,
            messages: List[Message],
            stream: bool = False
    ) -> AsyncGenerator[str, None]:
        try:
            # Get the user's latest query
            user_query = messages[-1].content

            # Get relevant context
            context, sources = await self._get_relevant_context(user_query)

            # Create the full prompt with context
            system_prompt = self._create_system_prompt(context)

            # Prepare messages for the LLM
            llm_messages = [
                {"role": "system", "content": system_prompt},
                *[{"role": m.role, "content": m.content} for m in messages]
            ]

            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=llm_messages,
                stream=stream,
                temperature=0.3,  # Lower temperature for more consistent regulatory responses
                max_tokens=2000
            )

            if stream:
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                response_text = response.choices[0].message.content
                # Add source citations if not streaming
                response_text += "\n\nSources:\n"
                for source in sources:
                    response_text += f"- {source['file_path']} ({source['file_type']})\n"
                yield response_text

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            error_message = (
                "I apologize, but I encountered an error processing your request. "
                "Please try again or contact IFSCA directly for assistance."
            )
            yield error_message


# FastAPI application
app = FastAPI(title="IFSCA Regulatory Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chat server
chat_server = ChatServer()


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        if request.stream:
            return StreamingResponse(
                chat_server.generate_response(request.messages, stream=True),
                media_type="text/event-stream"
            )
        else:
            response_text = ""
            sources = []
            async for chunk in chat_server.generate_response(request.messages, stream=False):
                response_text = chunk  # Will get the full response with sources

            # Extract sources from the response
            response_parts = response_text.split("\n\nSources:\n")
            main_response = response_parts[0]
            sources_text = response_parts[1] if len(response_parts) > 1 else ""

            # Parse sources
            sources = [
                {"source": line.strip("- ").split(" (")[0]}
                for line in sources_text.split("\n")
                if line.strip()
            ]

            return ChatResponse(
                response=main_response,
                sources=sources
            )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    try:
        # Verify Qdrant connection
        collections = chat_server.qdrant_store.client.get_collections()

        # Check if our collection exists
        collection_exists = any(
            c.name == COLLECTION_NAME
            for c in collections.collections
        )

        if not collection_exists:
            raise HTTPException(
                status_code=503,
                detail=f"Collection {COLLECTION_NAME} not found"
            )

        return {"status": "healthy", "collection": COLLECTION_NAME}

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Service unhealthy"
        )


if __name__ == "__main__":
    import uvicorn

    # Load environment variables
    if not os.getenv("DEEPSEEK_API_KEY"):
        logger.error("DEEPSEEK_API_KEY environment variable not set")
        raise EnvironmentError("DEEPSEEK_API_KEY not set")

    # Run the server
    logger.info("Starting chat server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )