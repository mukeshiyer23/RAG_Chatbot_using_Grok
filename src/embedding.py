import logging
from typing import List, Dict

from qdrant_client.http import models
from qdrant_client import QdrantClient

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


class EmbeddingProcessor:
    def __init__(self,
                 model_name='sentence-transformers/all-MiniLM-L6-v2',  # Full Hugging Face path
                 collection_prefix='regulatory_collection'):
        """
        Specialized embedding processor for regulatory documents
        """
        # Legal-specific embedding configuration
        self.embedding_model = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Vector Database Configuration
        self.qdrant_client = QdrantClient(
            host='localhost',
            port=6333,
            prefer_grpc=True
        )

        # Dynamic collection naming
        self.collection_name = f"{collection_prefix}_{self._generate_timestamp()}"

        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize vector collection
        self._create_specialized_collection()

    def _generate_timestamp(self):
        """Generate unique timestamp for collection"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _create_specialized_collection(self):
        """Create specialized Qdrant collection for regulatory documents"""
        try:
            # Aligned with model's actual output dimension
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # Matches new model's dimension
                    distance=models.Distance.COSINE
                )
            )
            self.logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Collection creation error: {e}")

    def _preprocess_regulatory_text(self, text: str) -> str:
        """
        Advanced text preprocessing for regulatory documents
        """
        text = text.lower()
        text = ' '.join(text.split())  # Remove extra whitespaces

        # Replace common regulatory abbreviations
        replacements = {
            'reg.': 'regulation',
            'sec.': 'section',
            'para.': 'paragraph',
            'subpara.': 'sub-paragraph'
        }
        for abbr, full in replacements.items():
            text = text.replace(abbr, full)

        return text

    def embed_text(self, text: str) -> List[float]:
        """
        Advanced embedding with regulatory domain preprocessing
        """
        preprocessed_text = self._preprocess_regulatory_text(text)

        embedding = self.embedding_model.encode(
            preprocessed_text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    async def upsert_regulatory_documents(self, documents: List[Dict]):
        """
        Batch upsert of regulatory documents
        """
        points = []
        for doc in documents:
            try:
                embedding = self.embed_text(doc['text'])

                point = models.PointStruct(
                    id=doc.get('id', hash(doc['text'])),
                    vector=embedding,
                    payload={
                        "text": doc['text'],
                        "filename": doc.get('filename', 'Unknown'),
                        "metadata": doc.get('metadata', {}),
                        "source_type": "regulatory_document"
                    }
                )
                points.append(point)
            except Exception as e:
                self.logger.error(f"Embedding error: {e}")

        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            self.logger.info(f"Upserted {len(points)} regulatory document points")

    async def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Semantic search with regulatory context
        """
        query_vector = self.embed_text(query)

        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )

        return [
            {
                "text": result.payload.get("text", ""),
                "filename": result.payload.get("filename", ""),
                "metadata": result.payload.get("metadata", {}),
                "similarity_score": result.score
            }
            for result in search_results
        ]