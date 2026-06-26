"""Qdrant multivector store- one multivector per page, ranked by MaxSim."""

from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from src.config import COLLECTION_NAME, QDRANT_PATH, TOP_K, VECTOR_DIM

_client: QdrantClient | None = None

def get_client() -> QdrantClient:
    """Open the on-disk Qdrant store once and reuse the connection."""
    global _client
    if _client is None:
        _client = QdrantClient(path=str(QDRANT_PATH))
    return _client


def close_client() -> None:
    """Close the local store explicitly so the file lock releases cleanly."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


def ensure_collection(reset: bool = False) -> None:
    """Create the multivector collection if it is missing."""
    client = get_client()
    if reset and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qm.VectorParams(
                size=VECTOR_DIM,
                distance=qm.Distance.COSINE,
                multivector_config=qm.MultiVectorConfig(
                    comparator=qm.MultiVectorComparator.MAX_SIM
                ),
            ),
        )


def upsert_page(point_id, multivector, pdf_name, page_number, image_path) -> None:
    """Store one page's multivector plus its source metadata."""
    client = get_client()
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            qm.PointStruct(
                id=point_id,
                vector=multivector,
                payload={"pdf": pdf_name, "page_number": page_number, "image_path": str(image_path)},
            )
        ],
    )