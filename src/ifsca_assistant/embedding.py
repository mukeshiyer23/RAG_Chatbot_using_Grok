# embedding.py (updated for production-grade use)
import os
import logging
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
import voyageai
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    def __init__(
            self,
            model_name: str = "voyage-law-2",
            collection_name: str = "ifsca_regulatory_docs",
            vector_size: int = 1024,
            distance_metric: str = "Cosine"
    ):
        """
        Initialize the embedding processor with Voyage AI and Qdrant.

        Args:
            model_name: Name of the Voyage AI embedding model.
            collection_name: Name of the Qdrant collection.
            vector_size: Dimension of the embedding vectors.
            distance_metric: Distance metric for vector search.
        """
        # Initialize Voyage AI client
        self.voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        self.model_name = model_name
        self.vector_size = vector_size

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
            prefer_grpc=True
        )
        self.collection_name = collection_name
        self.distance_metric = distance_metric

        # Create or validate collection
        self._initialize_collection()

    def _initialize_collection(self):
        """Ensure the Qdrant collection exists and is properly configured."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=self._get_distance_metric(self.distance_metric)
                    )
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise

    def _get_distance_metric(self, metric: str) -> models.Distance:
        """Map distance metric string to Qdrant's Distance enum."""
        metric_map = {
            "Cosine": models.Distance.COSINE,
            "Euclidean": models.Distance.EUCLID,
            "Dot": models.Distance.DOT
        }
        return metric_map.get(metric, models.Distance.COSINE)

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for text using Voyage AI.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        try:
            # Voyage AI embedding with document type
            result = self.voyage_client.embed(
                [text],
                model=self.model_name,
                input_type="document"
            )
            return result.embeddings[0]
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def semantic_search(self, query: str, top_k: int = 5, score_threshold: float = 0.7) -> List[Dict]:
        """
        Perform semantic search on the Qdrant collection.

        Args:
            query: Search query.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score for results.

        Returns:
            List of search results with text, metadata, and similarity score.
        """
        try:
            query_embedding = self.embed_text(query)

            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )

            return [
                {
                    "text": result.payload.get("text", ""),
                    "filename": result.payload.get("filename", "Unknown"),
                    "metadata": result.payload.get("metadata", {}),
                    "similarity_score": result.score
                }
                for result in search_results
            ]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise

    def upsert_points(self, points: List[Dict]):
        """
        Upsert points into the Qdrant collection.

        Args:
            points: List of points to upsert, each containing id, vector, and payload.
        """
        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point["id"],
                        vector=point["vector"],
                        payload=point["payload"]
                    )
                    for point in points
                ]
            )
            logger.info(f"Upserted {len(points)} points into collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to upsert points: {e}")
            raise

    def health_check(self) -> Dict:
        """Check the health of the embedding processor and its dependencies."""
        status = {
            "voyageai": False,
            "qdrant": False,
            "status": "unhealthy"
        }

        try:
            # Check Voyage AI
            self.voyage_client.ping()
            status["voyageai"] = True

            # Check Qdrant
            self.qdrant_client.get_collections()
            status["qdrant"] = True

            status["status"] = "healthy" if all(status.values()) else "degraded"
        except Exception as e:
            logger.error(f"Health check failed: {e}")

        return status