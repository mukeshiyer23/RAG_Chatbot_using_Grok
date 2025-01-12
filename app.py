import os

import numpy as np
from groq import Groq
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import glob
from collections import deque
from typing import List, Dict, Tuple
import logging
import json
from datetime import datetime
import hashlib
import spacy
import dateparser
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EnhancedRegulatoryBot:
    def __init__(self, api_key: str, data_folder: str = "data/", chunk_size: int = 512):
        self.client = Groq(api_key=api_key)
        self.data_folder = data_folder
        self.chunk_size = chunk_size
        self.context_history = deque(maxlen=5)
        self.documents_cache = None
        self.embedding_cache = {}
        self.embedding_cache_file = "embedding_cache.pt"

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('nli-roberta-base')
        self.cache_file = "document_cache.json"
        self.nlp = spacy.load('en_core_web_sm')
        self.last_query_context = None

        # Initialize documents and load embeddings
        self.initialize_system()

    def initialize_system(self):
        """Initialize the system with documents and embeddings."""
        # First load documents
        self.initialize_documents()

        # Then load or compute embeddings
        self.load_or_compute_embeddings()

    def load_or_compute_embeddings(self):
        """Load embeddings from cache or compute if necessary."""
        if os.path.exists(self.embedding_cache_file):
            try:
                self.embedding_cache = torch.load(self.embedding_cache_file)
                logging.info("Loaded embeddings from cache")
                return
            except Exception as e:
                logging.error(f"Error loading embedding cache: {e}")

        # If cache doesn't exist or is invalid, compute embeddings
        self.compute_and_cache_embeddings()

    def filter_chunks(self, chunks: List[Dict[str, str]], min_score: float = 0.3) -> List[Dict[str, str]]:
        return [chunk for chunk in chunks if chunk["score"] > min_score]

    def compute_and_cache_embeddings(self):
        """Compute embeddings for all chunks and cache them."""
        if not self.documents_cache:
            return

        all_chunks = []
        for doc in self.documents_cache:
            for chunk in doc["chunks"]:
                all_chunks.append(chunk)

        # Compute embeddings
        embeddings = self.embedding_model.encode(all_chunks, convert_to_tensor=True)

        # Cache embeddings
        self.embedding_cache = {
            'embeddings': embeddings,
            'chunks': all_chunks
        }

        # Save to disk
        torch.save(self.embedding_cache, self.embedding_cache_file)
        logging.info("Computed and cached embeddings")

    def _find_document_title(self, chunk_text: str) -> str:
        """Find the document title that contains this chunk of text."""
        for doc in self.documents_cache:
            if chunk_text in doc["chunks"]:
                return doc["title"]
        return "Unknown Source"

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """Retrieve relevant chunks using MMR (Maximal Marginal Relevance)."""
        try:
            # Encode query
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)

            # Get similarities
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                self.embedding_cache['embeddings'].cpu().numpy()
            )[0]

            # Convert similarities to tensor for consistent operations
            similarities = torch.from_numpy(similarities)

            # Initialize selection
            selected_indices = []
            remaining_indices = list(range(len(similarities)))

            while len(selected_indices) < top_k and remaining_indices:
                # Get relevance scores for remaining indices
                relevance_scores = similarities[remaining_indices]

                if not selected_indices:
                    # For first selection, just pick highest relevance
                    best_idx_pos = torch.argmax(relevance_scores)
                    best_idx = remaining_indices[best_idx_pos]
                else:
                    # Calculate diversity penalty
                    selected_embeddings = self.embedding_cache['embeddings'][selected_indices]
                    remaining_embeddings = self.embedding_cache['embeddings'][remaining_indices]

                    # Calculate diversity scores
                    diversity_matrix = 1 - torch.max(
                        cosine_similarity(
                            remaining_embeddings.cpu().numpy(),
                            selected_embeddings.cpu().numpy()
                        ),
                        axis=1
                    )
                    diversity_scores = torch.from_numpy(diversity_matrix)

                    # Combine relevance and diversity scores
                    lambda_param = 0.7
                    combined_scores = lambda_param * relevance_scores + (1 - lambda_param) * diversity_scores

                    # Select chunk with highest combined score
                    best_idx_pos = torch.argmax(combined_scores)
                    best_idx = remaining_indices[best_idx_pos]

                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

            # Return selected chunks with metadata
            results = []
            for idx in selected_indices:
                chunk_text = self.embedding_cache['chunks'][idx]
                # Find source document
                source_doc = None
                for doc in self.documents_cache:
                    if chunk_text in doc["chunks"]:
                        source_doc = doc
                        break

                results.append({
                    "text": chunk_text,
                    "title": source_doc["title"] if source_doc else "Unknown Source",
                    "score": float(similarities[idx])
                })

            return results

        except Exception as e:
            logging.error(f"Error in retrieve_relevant_chunks: {e}", exc_info=True)
            # Fallback to simple top-k selection if MMR fails
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                self.embedding_cache['embeddings'].cpu().numpy()
            )[0]

            top_indices = np.argsort(similarities)[-top_k:][::-1]

            return [
                {
                    "text": self.embedding_cache['chunks'][idx],
                    "title": next((doc["title"] for doc in self.documents_cache
                                   if self.embedding_cache['chunks'][idx] in doc["chunks"]),
                                  "Unknown Source"),
                    "score": float(similarities[idx])
                }
                for idx in top_indices
            ]

    def extract_metadata(self, text: str) -> Dict:
        """Extract key metadata from document text."""
        doc = self.nlp(text)
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        entities = [ent.text for ent in doc.ents if ent.label_ in {"ORG", "GPE"}]
        # Dummy categories and key topics as placeholders
        categories = ["General"]
        key_topics = [chunk.text for chunk in doc.noun_chunks][:5]

        return {
            "dates": dates,
            "entities": entities,
            "categories": categories,
            "key_topics": key_topics
        }

    def preprocess_query(self, query: str) -> Tuple[str, Dict]:
        """Enhanced query preprocessing with intent detection."""
        query_type = self._detect_query_type(query)
        entities = self._extract_entities(query)
        temporal_context = self._extract_temporal_context(query)

        context = {
            "query_type": query_type,
            "entities": entities,
            "temporal_context": temporal_context,
            "previous_context": self.last_query_context
        }

        return query, context

    def _detect_query_type(self, query: str) -> str:
        """Detect the type of regulatory query."""
        query_types = {
            "compliance": ["how to comply", "requirements", "mandatory"],
            "explanation": ["explain", "what is", "define"],
            "procedure": ["process", "steps", "how do I"],
            "updates": ["recent changes", "updates", "new regulations"]
        }

        query = query.lower()
        for qtype, keywords in query_types.items():
            if any(keyword in query for keyword in keywords):
                return qtype
        return "general"

    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from the query."""
        doc = self.nlp(query)
        return [ent.text for ent in doc.ents if ent.label_ in {"ORG", "GPE"}]

    def _extract_temporal_context(self, query: str) -> str:
        """Extract temporal context from the query."""
        parsed_date = dateparser.parse(query)
        return parsed_date.strftime("%Y-%m-%d") if parsed_date else ""

    def get_documents_hash(self) -> str:
        """Generate a hash of all documents in the data folder."""
        hash_md5 = hashlib.md5()
        for filepath in sorted(glob.glob(os.path.join(self.data_folder, "*.pdf"))):
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def initialize_documents(self):
        """Initialize documents with caching mechanism."""
        current_hash = self.get_documents_hash()

        # Try to load from cache first
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                if cache_data.get('hash') == current_hash:
                    logging.info("Loading documents from cache...")
                    self.documents_cache = cache_data['documents']
                    return
            except Exception as e:
                logging.error(f"Error loading cache: {e}")

        # If cache missing or invalid, process documents
        logging.info("Processing documents and creating new cache...")
        self.documents_cache = self.load_documents()

        # Save to cache
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'hash': current_hash,
                    'documents': self.documents_cache,
                    'timestamp': datetime.now().isoformat()
                }, f)
        except Exception as e:
            logging.error(f"Error saving cache: {e}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with improved formatting."""
        try:
            reader = PdfReader(pdf_path)
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(' '.join(page_text.split()))  # Normalize whitespace
            return "\n".join(text)
        except Exception as e:
            logging.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def load_documents(self) -> List[Dict[str, str]]:
        """Load documents with metadata."""
        texts = []
        file_paths = glob.glob(os.path.join(self.data_folder, "*.pdf"))

        for file_path in file_paths:
            text = self.extract_text_from_pdf(file_path)
            if text:
                filename = os.path.basename(file_path)
                name_without_ext = os.path.splitext(filename)[0]

                texts.append({
                    "text": text,
                    "file_path": file_path,
                    "filename": filename,
                    "title": name_without_ext.replace('_', ' ').title(),
                    "metadata": self.extract_metadata(text),
                    "chunks": self.split_into_chunks(text)
                })
                logging.info(f"Successfully loaded {filename}")

        return texts

    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks with better sentence awareness."""
        # Simple sentence boundary markers
        markers = ['. ', '? ', '! ', '\n\n']

        # First split into rough sentences
        sentences = []
        current = text
        for marker in markers:
            parts = []
            for part in current.split(marker):
                if part.strip():
                    parts.append(part.strip() + marker)
            current = ''.join(parts)
            sentences.extend(filter(None, current.split(marker)))

        # Then combine into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_size = 1000  # Larger chunks
        overlap = 200  # More overlap

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > chunk_size:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                # Start new chunk with some overlap
                overlap_point = max(0, len(current_chunk) - 3)  # Keep last ~3 sentences
                current_chunk = current_chunk[overlap_point:] if overlap_point > 0 else []
                current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def create_prompt(self, query: str, chunks: List[Dict[str, str]], query_context: Dict) -> str:
        # Format context text from relevant chunks
        context_text = "\n\n".join([
            f"Source: {chunk['title']}\n{chunk['text']}"
            for chunk in chunks
        ])

        # Simplified, more focused system prompt
        system_prompt = f"""You are an AI assistant for the International Financial Services Centres Authority (IFSCA). 
        Use the following information to provide clear, accurate answers about IFSCA regulations and guidelines.

        Current Query Information:
        - Type: {query_context.get('query_type', 'general')}
        - Related Entities: {', '.join(query_context.get('entities', ['None identified']))}
        - Time Context: {query_context.get('temporal_context', 'Current')}

        Relevant Documentation:
        {context_text}

        Query: {query}

        Provide a clear, direct response based on the provided documentation. If specific information isn't available 
        in the provided context, acknowledge this and provide general guidance based on available IFSCA regulations."""

        return system_prompt

    def generate_response(self, prompt: str, model: str = "llama3-70b-8192") -> str:
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system",
                     "content": "You are an AI assistant that only answers based on the provided context. If the context doesn't contain enough information to answer the question, say so directly."},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                temperature=0.3,  # Lower temperature for more focused responses
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "I apologize, but I couldn't find enough relevant information to answer your question accurately."

    def chat(self, query: str) -> Dict:
        """Enhanced chat interface with structured response."""
        try:
            processed_query, query_context = self.preprocess_query(query)
            relevant_chunks = self.retrieve_relevant_chunks(processed_query)

            # Add debug logging
            logging.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
            for i, chunk in enumerate(relevant_chunks):
                logging.info(f"Chunk {i}: {chunk['text'][:100]}...")  # Print first 100 chars

            prompt = self.create_prompt(processed_query, relevant_chunks, query_context)
            # Log the generated prompt
            logging.info(f"Generated prompt: {prompt[:500]}...")  # Print first 500 chars

            response = self.generate_response(prompt)
            logging.info(f"Raw response: {response}")

            self.last_query_context = query_context

            return {
                "status": "success",
                "answer": response,
                "sources": [chunk["title"] for chunk in relevant_chunks],
                "context": query_context
            }
        except Exception as e:
            logging.error(f"Error in chat pipeline: {e}", exc_info=True)
            return {
                "status": "error",
                "answer": "An error occurred while processing your request. Please try again.",
                "sources": [],
                "context": {}
            }


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set")
        return

    bot = EnhancedRegulatoryBot(api_key)
    print("Welcome to the Enhanced Regulatory Assistant! Type 'exit' to quit.")

    while True:
        try:
            query = input("\nYour question: ").strip()
            if query.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break

            response = bot.chat(query)
            print(f"\nStatus: {response['status']}")
            print(f"Assistant: {response['answer']}")

            if response['sources']:
                print("\nSources referenced:")
                for source in response['sources']:
                    print(f"- {source}")
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again.")


if __name__ == "__main__":
    main()
