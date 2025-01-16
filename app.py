import os
from groq import Groq
from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings
import asyncio
from typing import List, Dict, Any
import logging
import json
from datetime import datetime
import spacy
import dateparser
from tqdm import tqdm
import hashlib
from pathlib import Path
import aiofiles
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EnhancedRegulatoryBot:
    def __init__(self, api_key: str, collection_name: str = "data"):
        self.client = Groq(api_key=api_key)
        self.collection_name = collection_name
        self.nlp = spacy.load('en_core_web_sm')
        self.last_query_context = None

        # Initialize ChromaDB
        self.vector_store = chromadb.Client(Settings(
            persist_directory="data",  # Ensure "data" is the folder you are using for persistence
            anonymized_telemetry=False  # Optional: Turn off telemetry if needed
        ))

        # Get or create collection
        self.collection = self.vector_store.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    async def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process a single PDF file asynchronously"""
        try:
            reader = PdfReader(file_path)
            text_chunks = []

            # Process pages in chunks
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    # Clean and normalize text
                    text = ' '.join(text.split())
                    text_chunks.append(text)

            full_text = '\n'.join(text_chunks)
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]

            # Extract metadata
            metadata = await self.extract_metadata(full_text)

            return {
                "text": full_text,
                "file_path": file_path,
                "filename": filename,
                "title": name_without_ext.replace('_', ' ').title(),
                "metadata": metadata,
                "chunks": self.split_into_chunks(full_text)
            }
        except Exception as e:
            logging.error(f"Error processing PDF {file_path}: {e}")
            return None

    async def extract_metadata(self, text: str) -> Dict:
        """Extract metadata asynchronously"""
        # Run NLP processing in a thread pool
        with ThreadPoolExecutor() as executor:
            doc = await asyncio.get_event_loop().run_in_executor(
                executor, self.nlp, text
            )

        return {
            "dates": [ent.text for ent in doc.ents if ent.label_ == "DATE"],
            "entities": [ent.text for ent in doc.ents if ent.label_ in {"ORG", "GPE"}],
            "categories": ["General"],
            "key_topics": [chunk.text for chunk in doc.noun_chunks][:5]
        }

    def split_into_chunks(self, text: str, chunk_size: int = 512) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        overlap = 50  # Overlap words for context

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    async def initialize_documents(self, data_folder: str, batch_size: int = 100):
        """Initialize documents with batch processing"""
        try:
            pdf_files = list(Path(data_folder).glob("*.pdf"))

            for i in tqdm(range(0, len(pdf_files), batch_size)):
                batch = pdf_files[i:i + batch_size]

                # Process batch of PDFs concurrently
                tasks = [self.process_pdf(str(pdf)) for pdf in batch]
                documents = await asyncio.gather(*tasks)
                documents = [doc for doc in documents if doc]  # Remove None values

                if documents:
                    # Prepare data for ChromaDB
                    texts = []
                    ids = []
                    metadatas = []

                    for doc in documents:
                        for chunk in doc["chunks"]:
                            texts.append(chunk)
                            chunk_id = hashlib.md5(f"{doc['file_path']}-{chunk[:100]}".encode()).hexdigest()
                            ids.append(chunk_id)
                            metadatas.append({
                                "filename": doc["filename"],
                                "title": doc["title"],
                                "source": doc["file_path"],
                                **doc["metadata"]
                            })

                    # Add to ChromaDB
                    self.collection.add(
                        documents=texts,
                        ids=ids,
                        metadatas=metadatas
                    )

                    logging.info(f"Processed batch of {len(documents)} documents")

        except Exception as e:
            logging.error(f"Error initializing documents: {e}")
            raise

    async def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using ChromaDB"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            return [
                {
                    "text": doc,
                    "title": meta.get("title", "Unknown"),
                    "metadata": meta
                }
                for doc, meta in zip(
                    results["documents"][0],
                    results["metadatas"][0]
                )
            ]
        except Exception as e:
            logging.error(f"Error retrieving chunks: {e}")
            return []

    async def chat(self, query: str) -> Dict:
        """Enhanced chat interface with async processing"""
        try:
            processed_query, query_context = self.preprocess_query(query)
            relevant_chunks = await self.retrieve_relevant_chunks(processed_query)

            if not relevant_chunks:
                return {
                    "status": "no_context",
                    "answer": "No relevant information found. Please rephrase or provide more details.",
                    "sources": [],
                    "context": {}
                }

            prompt = self.create_prompt(processed_query, relevant_chunks, query_context)
            response = await self.generate_response(prompt)

            self.last_query_context = query_context

            return {
                "status": "success",
                "answer": response,
                "sources": [chunk["title"] for chunk in relevant_chunks],
                "context": query_context
            }
        except Exception as e:
            logging.error(f"Error in chat pipeline: {e}")
            return {
                "status": "error",
                "answer": "An error occurred while processing your request.",
                "sources": [],
                "context": {}
            }

    async def generate_response(self, prompt: str, model: str = "llama3-8b-8192") -> str:
        """Generate a response asynchronously"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt}
                ],
                model=model,
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "An error occurred while generating the response."


async def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set")
        return

    bot = EnhancedRegulatoryBot(api_key)

    # Initialize documents first
    data_folder = "data/"
    print("Initializing documents...")
    await bot.initialize_documents(data_folder)
    print("Document initialization complete!")

    print("\nWelcome to the Enhanced Regulatory Assistant! Type 'exit' to quit.")

    while True:
        try:
            query = input("\nYour question: ").strip()
            if query.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break

            response = await bot.chat(query)
            print(f"\nStatus: {response['status']}")
            print(f"Assistant: {response['answer']}")

            if response['sources']:
                print("\nSources:")
                for source in response['sources']:
                    print(f"- {source}")
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again.")


if __name__ == "__main__":
    asyncio.run(main())