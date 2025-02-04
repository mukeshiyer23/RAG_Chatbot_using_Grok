from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.utils.logger import logger

class QdrantStore:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self._verify_connection()

    def _verify_connection(self):
        try:
            self.client.get_collections()
            logger.info("Connected to Qdrant successfully")
        except Exception as e:
            logger.error(f"Qdrant connection failed: {str(e)}")
            raise

    def collection_exists(self, collection_name: str) -> bool:
        try:
            collections = self.client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except Exception as e:
            logger.error(f"Collection check failed: {str(e)}")
            raise

    def create_collection(
            self,
            collection_name: str,
            vector_size: int,
            recreate: bool = False
    ):
        try:
            if recreate and self.collection_exists(collection_name):
                self.client.delete_collection(collection_name)

            if not self.collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=2
                    ),
                )
                logger.info(f"Created collection: {collection_name}")
        except Exception as e:
            logger.error(f"Collection creation failed: {str(e)}")
            raise

    def add_embedding(
            self,
            collection_name: str,
            embedding: List[float],
            metadata: Dict,
            id: str = None
    ):
        try:
            point = models.PointStruct(
                id=id,
                vector=embedding,
                payload=metadata
            )
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            logger.info(f"Added embedding to {collection_name}")
        except Exception as e:
            logger.error(f"Failed to add embedding: {str(e)}")
            raise