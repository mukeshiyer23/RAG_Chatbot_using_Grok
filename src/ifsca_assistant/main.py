from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel
from typing import Optional, List
import os
import logging
import uuid
from dotenv import load_dotenv
from celery import Celery
import jwt
from datetime import datetime, timedelta

from .embedding import EmbeddingProcessor
from src.doc_handler import FileProcessor
from .services.llm_service import LLMService

# Configuration
load_dotenv()
app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# Initialize services
llm_service = LLMService()

# Celery configuration
celery = Celery(
    __name__,
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/1")
)

# Security
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
JWT_SECRET = os.getenv("JWT_SECRET", "supersecret")
ALGORITHM = "HS256"


# Rate limiting
@app.on_event("startup")
async def startup():
    await FastAPILimiter.init(redis_url=os.getenv("REDIS_URL"))


# Models
class ChatRequest(BaseModel):
    query: str


class User(BaseModel):
    username: str
    disabled: Optional[bool] = None


class Token(BaseModel):
    access_token: str
    token_type: str


# Authentication
def authenticate_user(username: str, password: str):
    # Implement proper user validation here
    return User(username=username)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)


# Endpoints
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials"
        )
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=120)
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/create-embedding")
async def create_embeddings(
        background_tasks: BackgroundTasks,
        current_user: User = Depends(RateLimiter(times=2, minutes=10))
):
    """Secure endpoint for initiating embedding creation"""
    try:
        background_tasks.add_task(process_documents_task.delay)
        return {"status": "queued", "task_id": str(uuid.uuid4())}
    except Exception as e:
        logger.error(f"Embedding creation error: {e}")
        raise HTTPException(status_code=500, detail="Embedding creation failed")


@app.post("/chat")
async def chat_endpoint(
        request: ChatRequest,
        current_user: User = Depends(RateLimiter(times=10, minutes=1))
):
    """Secure chat endpoint with rate limiting"""
    try:
        response = await handle_chat(request.query)
        return response
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat processing failed")


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    checks = {
        "qdrant": False,
        "voyageai": False,
        "deepseek": False,
        "redis": False
    }

    try:
        # Qdrant check
        EmbeddingProcessor().client.get_collections()
        checks["qdrant"] = True

        # Voyage check
        voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY")).ping()
        checks["voyageai"] = True

        # Deepseek check
        DeepseekClient(api_key=os.getenv("DEEPSEEK_API_KEY")).ping()
        checks["deepseek"] = True

        # Redis check
        celery.control.inspect().ping()
        checks["redis"] = True

        status_code = 200 if all(checks.values()) else 503
        return {"status": "healthy" if all(checks.values()) else "degraded", "checks": checks}

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "checks": checks}, 503


# Celery task
@celery.task(name="process_documents_task")
def process_documents_task():
    """Async document processing task"""
    try:
        processor = FileProcessor()
        processor.process_all_documents()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise


# Chat handling
async def handle_chat(query: str):
    """Enhanced chat handler with IFSCA-specific processing"""
    # Preprocess query
    cleaned_query = clean_query(query)
    if not cleaned_query:
        return {"answer": "Please provide a regulatory-related query for IFSCA."}

    # Retrieve context
    embedding_processor = EmbeddingProcessor()
    relevant_chunks = await embedding_processor.semantic_search(cleaned_query)

    # Generate response using Groq
    try:
        response = await llm_service.generate_response(cleaned_query, relevant_chunks)
        return {
            "answer": response,
            "sources": list(set([chunk['filename'] for chunk in relevant_chunks]))
        }
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return {"answer": "Unable to generate response", "sources": []}


def clean_query(query: str) -> str:
    """Clean and validate user queries"""
    query = query.strip().lower()

    # Handle greetings and empty queries
    greetings = {"hi", "hello", "hey"}
    if query in greetings:
        return "greeting"
    if len(query) < 3:
        return ""

    # Remove non-alphanumeric characters
    return ' '.join(word for word in query.split() if word.isalnum())


async def generate_ifsca_response(query: str, chunks: List[dict]):
    """IFSCA-specific response generation"""
    deepseek = DeepseekClient(api_key=os.getenv("DEEPSEEK_API_KEY"))

    if query == "greeting":
        return {
            "answer": "Welcome to the IFSCA Regulatory Assistant. How can I help you with financial sector "
                      "regulations today?",
            "sources": []
        }

    context = "\n".join([f"Source: {chunk['filename']}\n{chunk['text']}" for chunk in chunks[:3]])

    prompt = f"""You are a regulatory expert assistant for IFSCA (International Financial Services Centres Authority). 
    Use the following context to answer the query. If unsure, state you don't know.

    Context:
    {context}

    Query: {query}

    Provide a concise, accurate response citing relevant regulations when possible.
    """

    try:
        response = deepseek.generate(
            model="deepseek-r1",
            prompt=prompt,
            max_tokens=500,
            temperature=0.2
        )
        return {
            "answer": response.text,
            "sources": list(set([chunk['filename'] for chunk in chunks]))
        }
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return {"answer": "Unable to generate response", "sources": []}