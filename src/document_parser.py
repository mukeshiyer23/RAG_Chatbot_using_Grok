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

import pytesseract
import langdetect
import cv2

import pandas as pd  # Added for XLSX support
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
            data = json.load(file)
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Intelligent text chunking for semantic search
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def preprocess_image_for_ocr(image_path):
    """Preprocess image to improve OCR accuracy"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    denoised = cv2.fastNlMeansDenoising(thresh)
    return denoised


def extract_text_from_scanned_pdf(pdf_path):
    """Extract text from scanned PDFs using OCR"""
    full_text = ""
    reader = PdfReader(pdf_path)
    os.makedirs('temp_pdf_images', exist_ok=True)

    for i, page in enumerate(reader.pages):
        try:
            if '/XObject' in page['/Resources']:
                xObject = page['/Resources']['/XObject'].get_object()

                for obj in xObject:
                    if xObject[obj]['/Subtype'] == '/Image':
                        size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                        data = xObject[obj].get_data()

                        img_path = f'temp_pdf_images/page_{i}.png'
                        with open(img_path, 'wb') as img_file:
                            img_file.write(data)

                        preprocessed_img = preprocess_image_for_ocr(img_path)
                        cv2.imwrite(img_path, preprocessed_img)

                        page_text = pytesseract.image_to_string(img_path)
                        full_text += page_text + "\n\n"
        except Exception as e:
            logging.error(f"OCR error on page {i}: {e}")

    import shutil
    shutil.rmtree('temp_pdf_images', ignore_errors=True)

    return full_text


def filter_english_text(text):
    """Filter out non-English text"""
    potential_chunks = text.split('\n')
    english_chunks = []

    for chunk in potential_chunks:
        try:
            if langdetect.detect(chunk) == 'en':
                english_chunks.append(chunk)
        except langdetect.lang_detect_exception.LangDetectException:
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

        for para in document.paragraphs:
            text = para.text.strip()
            if text:
                chunks.append(text)

        return chunks

    async def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process PDFs with robust text extraction"""
        try:
            reader = PdfReader(file_path)
            full_text = ""

            for page in reader.pages:
                page_text = page.extract_text() or ""
                full_text += page_text + "\n\n"

            if not full_text.strip():
                full_text = extract_text_from_scanned_pdf(file_path)

            full_text = _clean_text(full_text)
            full_text = filter_english_text(full_text)

            chunks = split_into_chunks(full_text)

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

        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'li']):
            text = element.get_text(strip=True)
            if text:
                chunks.append(text)

        return chunks

    async def process_markdown(self, file_path: str) -> Dict[str, Any]:
        """Process Markdown files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                full_text = file.read()

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

    async def process_xlsx(self, file_path: str) -> Dict[str, Any]:
        """Process Excel files with comprehensive text extraction"""
        try:
            # Read all sheets
            xls = pd.ExcelFile(file_path)
            full_text = []

            for sheet_name in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)

                # Convert DataFrame to text, handling different data types
                sheet_text = []
                for col in df.columns:
                    column_text = f"Column: {col}\n"
                    column_text += "\n".join(df[col].astype(str).dropna())
                    sheet_text.append(column_text)

                full_text.append(f"Sheet: {sheet_name}\n" + "\n".join(sheet_text))

            full_text_str = "\n\n".join(full_text)
            chunks = split_into_chunks(full_text_str)

            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]

            return {
                "filename": filename,
                "title": name_without_ext.replace('_', ' ').title(),
                "chunks": chunks,
                "metadata": extract_dict_from_json(file_path + ".json")
            }
        except Exception as e:
            logging.error(f"Error processing Excel {file_path}: {e}")
            return None

    async def parse_documents(self):
        # Extended file type support
        document_files = []
        for ext in ['.pdf', '.docx', '.html', '.md', '.txt', '.xlsx']:
            document_files.extend(list(Path(self.data_folder).rglob(f'*{ext}')))

        print(f"Total files found: {len(document_files)}")

        for i in tqdm(range(0, len(document_files), self.batch_size)):
            batch = document_files[i:i + self.batch_size]

            async def process_file(file_path):
                try:
                    if file_path.suffix.lower() == '.pdf':
                        document = await self.process_pdf(str(file_path))
                    elif file_path.suffix.lower() == '.docx':
                        document = await self.process_docx(str(file_path))
                    elif file_path.suffix.lower() == '.html':
                        document = await self.process_html(str(file_path))
                    elif file_path.suffix.lower() in ['.md', '.txt']:
                        document = await self.process_markdown(str(file_path))
                    elif file_path.suffix.lower() == '.xlsx':
                        document = await self.process_xlsx(str(file_path))
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
                        chunk_id = hashlib.md5(f"{file_path}-{chunk[:100]}".encode()).hexdigest()

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