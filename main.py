import os
from pathlib import Path
import uuid
import json
from typing import Dict

from src.doc_handler import PdfHandler, DocxHandler, ExcelHandler, MarkdownHandler
from src.embedder import Embedder
from src.qdrant_store import QdrantStore
from src.utils.logger import logger

COLLECTION_NAME = "documents"
HANDLERS: Dict[str, callable] = {
    ".pdf": PdfHandler,
    ".docx": DocxHandler,
    ".xlsx": ExcelHandler,
    ".xls": ExcelHandler,
    ".md": MarkdownHandler,
}


def load_metadata(file_path: Path) -> dict:
    meta_path = file_path.with_suffix(file_path.suffix + '.meta.json')
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata from {meta_path}: {str(e)}")
    return {}


def process_file(file_path: Path, embedder: Embedder, qdrant_store: QdrantStore):
    try:
        if file_path.suffix == '.json':
            return

        file_extension = file_path.suffix.lower()
        if file_extension not in HANDLERS:
            logger.warning(f"Unsupported file type: {file_path}")
            return

        metadata = load_metadata(file_path)
        handler_class = HANDLERS[file_extension]
        handler = handler_class(str(file_path))
        handler.metadata = metadata

        logger.info(f"Extracting content from {file_path}")
        handler.extract_content()

        logger.info(f"Cleaning content from {file_path}")
        handler.clean_content()

        if not handler.cleaned_content.strip():
            logger.warning(f"No valid content found in {file_path} after cleaning")
            return

        logger.info(f"Generating embeddings for {file_path}")
        chunks = handler.chunk_text(handler.cleaned_content)

        for i, chunk in enumerate(chunks):
            embedding = embedder.embed(chunk)
            chunk_metadata = {
                "file_path": str(file_path),
                "file_type": file_extension,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "content": chunk,
                **metadata
            }

            qdrant_store.add_embedding(
                collection_name=COLLECTION_NAME,
                embedding=embedding,
                metadata=chunk_metadata,
                id=str(uuid.uuid4())
            )

        logger.info(f"Successfully processed {file_path}")

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {str(e)}")


def process_directory(directory_path: str, embedder: Embedder, qdrant_store: QdrantStore):
    try:
        directory = Path(directory_path)
        for file_path in directory.rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                process_file(file_path, embedder, qdrant_store)
    except Exception as e:
        logger.error(f"Failed to process directory {directory_path}: {str(e)}")


if __name__ == "__main__":
    logger.info("Initializing embedder and Qdrant store...")
    embedder = Embedder()
    qdrant_store = QdrantStore()

    qdrant_store.create_collection(
        collection_name=COLLECTION_NAME,
        vector_size=embedder.vector_size,
        recreate=True
    )

    input_directory = "data"
    logger.info(f"Processing files in directory: {input_directory}")
    process_directory(input_directory, embedder, qdrant_store)
    logger.info("Processing completed.")