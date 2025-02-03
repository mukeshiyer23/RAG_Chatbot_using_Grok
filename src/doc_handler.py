import json
import os
import re
import concurrent.futures
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Callable

import pandas as pd
from langdetect import DetectorFactory, detect, LangDetectException

from tika import parser as tika_parser
from docx import Document
from PyPDF2 import PdfReader
import pytesseract
from pdf2image import convert_from_path

from src.embedder import Embedder
from src.qdrant_store import QdrantStore
from src.utils import utils
from src.utils.exceptions import DocumentProcessingError, OCRProcessingError, \
    LanguageDetectionError
from src.utils.logger import logger

# Initialize detector factory for consistent language detection
DetectorFactory.seed = 0


class DocHandler(ABC):
    def __init__(
            self,
            file_path: str,
            custom_cleaners: Optional[List[Callable[[str], str]]] = None,
            clean_patterns: Optional[List[str]] = None
    ):
        self.file_path = Path(file_path)
        self.meta_path = self.file_path.with_suffix(self.file_path.suffix + '.meta.json')
        self.metadata = self._load_metadata()
        self.raw_content = ""
        self.cleaned_content = ""  # Renamed from clean_content to avoid conflict
        self.custom_cleaners = custom_cleaners or []
        self.clean_patterns = clean_patterns or [r'\s+']

        if not self.file_path.exists():
            logger.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def _load_metadata(self) -> Dict:
        try:
            if not self.meta_path.exists():
                logger.warning(f"No metadata found for {self.file_path}")
                return {}

            with open(self.meta_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid metadata JSON for {self.file_path}: {str(e)}")
            return {}

    @abstractmethod
    def extract_content(self):
        pass

    def clean_content(self):
        try:
            # Apply basic cleaning patterns
            cleaned = self.raw_content
            for pattern in self.clean_patterns:
                cleaned = re.sub(pattern, ' ', cleaned)

            # Apply custom cleaning functions
            for cleaner in self.custom_cleaners:
                cleaned = cleaner(cleaned)

            self.cleaned_content = cleaned.strip()  # Using renamed variable

            # Language validation
            if not self.is_english(self.cleaned_content):
                logger.warning(f"Non-English content detected in {self.file_path}")
                self.cleaned_content = ""
        except LangDetectException as e:
            logger.error(f"Language detection failed for {self.file_path}: {str(e)}")
            raise LanguageDetectionError("Language detection failed") from e
        except Exception as e:
            logger.error(f"Cleaning failed for {self.file_path}: {str(e)}")
            raise DocumentProcessingError("Content cleaning failed") from e

    def is_english(self, text: str) -> bool:
        try:
            return detect(text) == 'en'
        except LangDetectException as e:
            logger.error(f"Language detection error: {str(e)}")
            raise LanguageDetectionError("Language detection failed") from e

    def chunk_text(
            self,
            text: str,
            chunk_size: int = 512,
            overlap: int = 50
    ) -> List[str]:
        try:
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
            chunks = []
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                words = sentence.split()
                sentence_length = len(words)

                if current_length + sentence_length > chunk_size:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = current_chunk[-overlap:] + words
                    current_length = sum(len(word) for word in current_chunk)
                else:
                    current_chunk.extend(words)
                    current_length += sentence_length

            if current_chunk:
                chunks.append(' '.join(current_chunk))

            return chunks
        except Exception as e:
            logger.error(f"Chunking failed for {self.file_path}: {str(e)}")
            raise DocumentProcessingError("Text chunking failed") from e


class PdfHandler(DocHandler):
    def __init__(self, *args, ocr_fallback: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.ocr_fallback = ocr_fallback

    def extract_content(self):
        try:
            # First try text extraction
            parsed = tika_parser.from_file(str(self.file_path))
            self.raw_content = parsed['content'] or ""

            # Fallback to PyPDF2 if Tika fails
            if not self.raw_content.strip():
                reader = PdfReader(self.file_path)
                self.raw_content = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )

            # OCR fallback if still no content
            if self.ocr_fallback and not self.raw_content.strip():
                self._perform_ocr()
        except Exception as e:
            logger.error(f"PDF extraction failed for {self.file_path}: {str(e)}")
            if self.ocr_fallback:
                self._perform_ocr()
            else:
                raise DocumentProcessingError("PDF processing failed") from e

    def _perform_ocr(self):
        try:
            logger.info(f"Attempting OCR for {self.file_path}")
            images = convert_from_path(self.file_path)
            ocr_text = []

            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                ocr_text.append(f"Page {i + 1}:\n{text}")

            self.raw_content = "\n\n".join(ocr_text)

            if not self.raw_content.strip():
                raise OCRProcessingError("OCR produced empty content")
        except Exception as e:
            logger.error(f"OCR failed for {self.file_path}: {str(e)}")
            raise OCRProcessingError("OCR processing failed") from e


class DocxHandler(DocHandler):
    def extract_content(self):
        try:
            doc = Document(str(self.file_path))
            self.raw_content = "\n".join(
                para.text for para in doc.paragraphs if para.text.strip()
            )
        except Exception as e:
            logger.error(f"DOCX extraction failed for {self.file_path}: {str(e)}")
            raise DocumentProcessingError("DOCX processing failed") from e


class ExcelHandler(DocHandler):
    def extract_content(self):
        try:
            df = pd.read_excel(self.file_path)
            self.raw_content = df.to_string(index=False)
        except Exception as e:
            logger.error(f"Excel extraction failed for {self.file_path}: {str(e)}")
            raise DocumentProcessingError("Excel processing failed") from e


class MarkdownHandler(DocHandler):
    def extract_content(self):
        """Extract content with frontmatter handling"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Handle frontmatter
            if content.startswith('---'):
                # Find the second '---' to extract the main content
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    self.raw_content = parts[2].strip()
                    self.metadata = utils.convert_to_json(parts[1])
                else:
                    self.raw_content = content
            else:
                self.raw_content = content

        except Exception as e:
            logger.error(f"Markdown extraction failed for {self.file_path}: {str(e)}")
            raise DocumentProcessingError("Markdown processing failed") from e

    def clean_content(self):
        """Enhanced cleaning with frontmatter and markdown syntax handling"""
        try:
            if not self.raw_content.strip():
                raise DocumentProcessingError("Empty raw content before cleaning")

            cleaned = self.raw_content

            # Remove markdown syntax
            cleaned = re.sub(r'#{1,6}\s*(.*?)\n', r'\1\n', cleaned)  # Headers
            cleaned = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cleaned)  # Links
            cleaned = re.sub(r'(?<!\*)\*{1,2}([^*]+)\*{1,2}(?!\*)', r'\1', cleaned)  # Bold/Italic
            cleaned = re.sub(r'`[^`]+`', '', cleaned)  # Inline code
            cleaned = re.sub(r'```[\s\S]*?```', '', cleaned)  # Code blocks
            cleaned = re.sub(r'>\s*(.*?)\n', r'\1\n', cleaned)  # Blockquotes
            cleaned = re.sub(r'^\s*[-*+]\s+', '', cleaned, flags=re.MULTILINE)  # List items
            cleaned = re.sub(r'^\s*\d+\.\s+', '', cleaned, flags=re.MULTILINE)  # Numbered lists

            # Apply basic cleaning patterns
            for pattern in self.clean_patterns:
                cleaned = re.sub(pattern, ' ', cleaned)

            # Apply custom cleaning functions
            for cleaner in self.custom_cleaners:
                cleaned = cleaner(cleaned)

            # Final cleanup
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
            cleaned = cleaned.strip()

            # Validate content length and structure
            if not self._validate_content(cleaned):
                raise DocumentProcessingError("Content validation failed after cleaning")

            self.cleaned_content = cleaned

        except Exception as e:
            logger.error(f"Cleaning failed for {self.file_path}: {str(e)}")
            raise DocumentProcessingError("Content cleaning failed") from e

    def _validate_content(self, text: str) -> bool:
        """Validate content quality and length"""
        if not text.strip():
            return False

        # Check for minimum content length (adjust thresholds as needed)
        words = re.findall(r'\b\w+\b', text)
        if len(words) < 10:
            logger.warning(f"Insufficient content length in {self.file_path}")
            return False

        # Check for basic text structure
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) > 3]
        if not valid_sentences:
            logger.warning(f"No valid sentences found in {self.file_path}")
            return False

        return True

    def is_english(self, text: str) -> bool:
        """Improved language detection with better error handling"""
        try:
            # Pre-process text for language detection
            text = text.strip()

            # Basic validation
            if not text or len(text) < 20:
                logger.warning(f"Text too short for language detection in {self.file_path}")
                return True  # Assume English for very short texts

            # Remove common markdown artifacts that might interfere
            text = re.sub(r'[#*`_-]', '', text)

            # Get a sample of the text for language detection
            # Use the first 1000 characters that contain actual words
            words_text = ' '.join(re.findall(r'\b\w+\b', text))[:1000]

            if not words_text:
                logger.warning(f"No valid words found for language detection in {self.file_path}")
                return True  # Assume English if no valid words (likely structured data)

            return detect(words_text) == 'en'

        except LangDetectException as e:
            logger.warning(f"Language detection failed for {self.file_path}, assuming English: {str(e)}")
            return True  # Assume English on detection failure
        except Exception as e:
            logger.error(f"Unexpected error in language detection: {str(e)}")
            return True  # Assume English on unexpected errors