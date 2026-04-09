import torch
from PIL import Image
from qdrant_client import QdrantClient
from app.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    TOP_K
)


def get_qdrant_client():
    return QdrantClient(path="data/qdrant_local")

def embed_query(query, model, processor):
    with torch.no_grad():
        batch = processor.process_queries([query]).to(model.device)
        embedding = model(**batch)
    return embedding[0].cpu().float().numpy().tolist()


def retrieve_pages(query, model, processor, client, top_k=TOP_K):
    print(f"Searching for: '{query}'")
    query_vector = embed_query(query, model, processor)

    results = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )

    retrieved = []
    for point in results.points:
        payload = point.payload
        image_path = payload.get("image_path")
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"  Could not load image {image_path}: {e}")
            image = None

        retrieved.append({
            "doc_name": payload.get("doc_name"),
            "page_number": payload.get("page_number"),
            "image_path": image_path,
            "image": image,
            "score": point.score
        })
        print(f"  → {payload.get('doc_name')} — Page {payload.get('page_number')} (score: {point.score:.3f})")

    return retrieved