import asyncio
import json
import logging
import os
from typing import List, Dict, Any

from docx import Document
from tqdm import tqdm
from bs4 import BeautifulSoup

from PyPDF2 import PdfReader
import re
from pathlib import Path
import hashlib

from qdrant_client.http import models

import pytesseract
import langdetect
import cv2

from src.embedding import EmbeddingProcessor


def _clean_text(text: str) -> str:
    """Clean and normalize text for processing"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)

    return text


def extract_dict_from_json(json_path):
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)  # Load JSON content as a Python dictionary
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Intelligent text chunking for semantic search

    Key features:
    - Preserve semantic context
    - Handle variable-length documents
    - Maintain readability
    """
    # Split into words to ensure semantic integrity
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        # Create chunk with overlap
        chunk = ' '.join(words[i:i + chunk_size])

        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def preprocess_image_for_ocr(image_path):
    """
    Preprocess image to improve OCR accuracy

    Steps:
    - Convert to grayscale
    - Apply adaptive thresholding
    - Denoise
    - Deskew
    """
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)

    return denoised


def extract_text_from_scanned_pdf(pdf_path):
    """
    Extract text from scanned PDFs using OCR
    """
    full_text = ""

    # Open PDF
    reader = PdfReader(pdf_path)

    # Temporary directory for image extraction
    os.makedirs('temp_pdf_images', exist_ok=True)

    for i, page in enumerate(reader.pages):
        try:
            # Extract image from PDF page
            if '/XObject' in page['/Resources']:
                xObject = page['/Resources']['/XObject'].get_object()

                for obj in xObject:
                    if xObject[obj]['/Subtype'] == '/Image':
                        size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                        data = xObject[obj].get_data()

                        # Save image
                        img_path = f'temp_pdf_images/page_{i}.png'
                        with open(img_path, 'wb') as img_file:
                            img_file.write(data)

                        # Preprocess image
                        preprocessed_img = preprocess_image_for_ocr(img_path)
                        cv2.imwrite(img_path, preprocessed_img)

                        # Extract text with Tesseract
                        page_text = pytesseract.image_to_string(img_path)
                        full_text += page_text + "\n\n"
        except Exception as e:
            logging.error(f"OCR error on page {i}: {e}")

    # Clean up temporary images
    import shutil
    shutil.rmtree('temp_pdf_images', ignore_errors=True)

    return full_text


def filter_english_text(text):
    """
    Filter out non-English text
    Returns only English text chunks
    """
    # Split into potential chunks
    potential_chunks = text.split('\n')
    english_chunks = []

    for chunk in potential_chunks:
        try:
            # Detect language
            if langdetect.detect(chunk) == 'en':
                english_chunks.append(chunk)
        except langdetect.lang_detect_exception.LangDetectException:
            # If language detection fails, skip the chunk
            continue

    return '\n'.join(english_chunks)


class DocumentParser:
    def __init__(self, embedding: EmbeddingProcessor):
        self.data_folder = "data"
        self.batch_size = 50
        self.embedding = embedding
    async def process_docx(self, docx_path: str):
        """Extracts text chunks from a DOCX file."""
        document = Document(docx_path)
        chunks = []

        # Extract paragraphs and add them as chunks
        for para in document.paragraphs:
            text = para.text.strip()
            if text:  # Skip empty paragraphs
                chunks.append(text)

        return chunks

    async def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process PDFs with robust text extraction"""
        try:
            reader = PdfReader(file_path)
            full_text = ""

            # Extract text from all pages
            for page in reader.pages:
                # Use page.extract_text() directly without indexing
                page_text = page.extract_text() or ""
                full_text += page_text + "\n\n"

            # Check if PDF is effectively blank or scanned
            if not full_text.strip():
                # Fallback to OCR method
                full_text = extract_text_from_scanned_pdf(file_path)

            # Clean and normalize text
            full_text = _clean_text(full_text)

            # Filter for English text
            full_text = filter_english_text(full_text)

            # Break into manageable chunks
            chunks = split_into_chunks(full_text)

            # Extract metadata
            metadata_path = file_path + ".json"
            metadata = extract_dict_from_json(metadata_path)

            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]

            return {
                "text": full_text,
                "file_path": file_path,
                "filename": filename,
                "title": name_without_ext.replace('_', ' ').title(),
                "chunks": chunks,
                "metadata": metadata
            }
        except Exception as e:
            logging.error(f"PDF processing error for {file_path}: {e}")
            return None

    async def process_html(self, html_path: str):
        """Extracts text chunks from an HTML file."""
        with open(html_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')

        chunks = []

        # Extract text from all paragraphs, headers, and other textual elements
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'li']):
            text = element.get_text(strip=True)
            if text:
                chunks.append(text)

        return chunks

    async def process_markdown(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                full_text = file.read()

            # Enhanced markdown parsing
            chunks = split_into_chunks(full_text, chunk_size=500, overlap=50)

            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]

            return {
                "filename": filename,
                "title": name_without_ext.replace('_', ' ').title(),
                "chunks": chunks,
                "metadata": extract_dict_from_json(file_path + ".json")
            }
        except Exception as e:
            logging.error(f"Error processing Markdown {file_path}: {e}")
            return None

    async def parse_documents(self):
        # Find all document files recursively
        document_files = []
        for ext in ['.pdf', '.docx', '.html', '.md', '.txt']:
            document_files.extend(list(Path(self.data_folder).rglob(f'*{ext}')))

        print(f"Total files found: {len(document_files)}")

        # Process files in batches
        for i in tqdm(range(0, len(document_files), self.batch_size)):
            batch = document_files[i:i + self.batch_size]

            async def process_file(file_path):
                try:
                    # Determine file type and process accordingly
                    if file_path.suffix.lower() == '.pdf':
                        document = {
                            'filename': file_path.name,
                            'title': file_path.stem,
                            'chunks': await self.process_pdf(str(file_path)),
                            'metadata': extract_dict_from_json(file_path.name + ".json")
                        }
                    elif file_path.suffix.lower() == '.docx':
                        document = {
                            'filename': file_path.name,
                            'title': file_path.stem,
                            'chunks': await self.process_docx(str(file_path)),
                            'metadata': extract_dict_from_json(file_path.name + ".json")
                        }
                    elif file_path.suffix.lower() == '.html':
                        document = {
                            'filename': file_path.name,
                            'title': file_path.stem,
                            'chunks': await self.process_html(str(file_path)),
                            'metadata': extract_dict_from_json(file_path.name + ".json")
                        }
                    elif file_path.suffix.lower() in ['.md', '.txt']:
                        document = {
                            'filename': file_path.name,
                            'title': file_path.stem,
                            'chunks': await self.process_markdown(str(file_path)),
                            'metadata': extract_dict_from_json(file_path.name + ".json")
                        }
                    else:
                        print(f"Unsupported file type: {file_path}")
                        return None

                    return document
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    return None

            tasks = [process_file(file_path) for file_path in batch]
            documents = await asyncio.gather(*tasks)

            documents = [doc for doc in documents if doc]

            if documents:
                points = []
                for doc in documents:
                    metadata = doc.get('metadata', {})
                    file_path = doc.get('file_path', 'Unknown')
                    filename = doc.get('filename', 'Untitled')
                    title = doc.get('title', 'Untitled')

                    for chunk in doc.get("chunks", []):
                        # Generate unique chunk ID
                        chunk_id = hashlib.md5(f"{file_path}-{chunk[:100]}".encode()).hexdigest()

                        # Explicitly generate embedding for each chunk
                        try:
                            chunk_embedding = self.embedding.embed_text(chunk)

                            points.append(
                                models.PointStruct(
                                    id=chunk_id,
                                    vector=chunk_embedding,
                                    payload={
                                        "text": chunk,
                                        "filename": filename,
                                        "title": title,
                                        **metadata
                                    }
                                )
                            )

                            print(f"Embedding created for chunk from {filename}")
                        except Exception as e:
                            print(f"Embedding failed for chunk from {filename}: {e}")

                if points:
                    try:
                        self.embedding.qdrant_client.upsert(
                            collection_name=self.embedding.collection_name,
                            points=points
                        )
                        print(f"Successfully upserted {len(points)} embedding points")
                    except Exception as e:
                        logging.error(f"Error upserting points: {e}")
