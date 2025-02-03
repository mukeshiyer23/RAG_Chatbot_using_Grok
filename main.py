import os
from pathlib import Path

from src.doc_handler import PdfHandler, DocxHandler, ExcelHandler, MarkdownHandler
from src.embedder import Embedder
from src.qdrant_store import QdrantStore
from src.utils.exceptions import UnsupportedFileTypeError
from src.utils.logger import logger

from qdrant_client.http import models

import concurrent.futures


def process_file(file_path: Path, embedder: Embedder, qdrant_store: QdrantStore):
    try:
        handler = None
        if file_path.suffix.lower() == '.pdf':
            handler = PdfHandler(file_path)
        elif file_path.suffix.lower() == '.docx':
            handler = DocxHandler(file_path)
        elif file_path.suffix.lower() in ('.xlsx', '.xls'):
            handler = ExcelHandler(file_path)
        elif file_path.suffix.lower() == '.md':
            handler = MarkdownHandler(file_path)
        else:
            raise UnsupportedFileTypeError(f"Unsupported file type: {file_path.suffix}")

        handler.extract_content()
        handler.clean_content()

        if not handler.clean_content:
            logger.warning(f"Skipping empty content in {file_path}")
            return

        chunks = handler.chunk_text(handler.cleaned_content)
        points = []
        for idx, chunk in enumerate(chunks):
            embedding = embedder.embed(chunk)
            points.append(models.PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": str(file_path),
                    "metadata": handler.metadata
                }
            ))

        qdrant_store.save_embedding("documents", points)
        logger.info(f"Processed {file_path} successfully")

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {str(e)}")


def process_documents(
        data_dir: str = "data",
        num_workers: int = 4,
        collection_name: str = "documents"
):
    try:
        embedder = Embedder()
        qdrant_store = QdrantStore()

        # Collection management
        qdrant_store.create_collection(
            collection_name=collection_name,
            vector_size=embedder.model.get_sentence_embedding_dimension(),
            recreate=False
        )

        # Gather files to process
        file_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.meta.json'):
                    continue
                file_path = Path(root) / file
                file_paths.append(file_path)

        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    process_file,
                    file_path,
                    embedder,
                    qdrant_store
                )
                for file_path in file_paths
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Processing failed: {str(e)}")

        logger.info("Document processing completed")

    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")


process_documents(data_dir="data", num_workers=2)
