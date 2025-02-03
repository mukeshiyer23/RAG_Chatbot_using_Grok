class DocumentProcessingError(Exception):
    """Base class for document processing exceptions"""


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when encountering unsupported file types"""


class OCRProcessingError(DocumentProcessingError):
    """Raised when OCR processing fails"""


class LanguageDetectionError(DocumentProcessingError):
    """Raised when language detection fails"""

