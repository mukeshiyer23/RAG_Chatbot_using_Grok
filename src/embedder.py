from sentence_transformers import SentenceTransformer
from src.utils.logger import logger


class Embedder:
    def __init__(self, model_name: str = 'BAAI/bge-large-en'):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
            self.vector_size = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

    def embed(self, text: str):
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            raise
