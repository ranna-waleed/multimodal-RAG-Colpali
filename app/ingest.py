import torch
import fitz
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client import models
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from app.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    COLPALI_MODEL_NAME,
    DPI
)
import os
import uuid


def load_model():
    print("Loading ColIdefics3 model...")
    device = "cpu"
    model = ColIdefics3.from_pretrained(
        COLPALI_MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=device
    ).eval()
    processor = ColIdefics3Processor.from_pretrained(COLPALI_MODEL_NAME)
    print(f"Model loaded on {device}")
    return model, processor


def get_qdrant_client():
    return QdrantClient(path="data/qdrant_local")

def setup_collection(client):
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION_NAME not in existing:
        print(f"Creating collection: {QDRANT_COLLECTION_NAME}")
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                )
            )
        )
        print("Collection created")
    else:
        print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists")


def pdf_to_images(pdf_path):
    print(f"Converting PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(DPI / 72, DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    print(f"  → {len(images)} pages converted")
    return images


def save_page_images(images, doc_name, save_dir="data/page_images"):
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    for i, img in enumerate(images):
        path = os.path.join(save_dir, f"{doc_name}_page_{i+1}.jpg")
        img.save(path, "JPEG")
        saved_paths.append(path)
    return saved_paths


def ingest_pdf(pdf_path, model, processor, client):
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    images = pdf_to_images(pdf_path)
    saved_paths = save_page_images(images, doc_name)

    print("Generating embeddings...")
    points = []
    for i, image in enumerate(images):
        print(f"  → Embedding page {i+1}/{len(images)}")
        with torch.no_grad():
            batch = processor.process_images([image]).to(model.device)
            embedding = model(**batch)

        vector = embedding[0].cpu().float().numpy().tolist()
        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "doc_name": doc_name,
                "page_number": i + 1,
                "image_path": saved_paths[i],
                "total_pages": len(images)
            }
        )
        points.append(point)

    # Upload in batches of 10 pages
    batch_size = 10
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=batch)
        print(f"  → Uploaded pages {i+1} to {min(i+batch_size, len(points))}")

    print(f" Ingested '{doc_name}' — {len(images)} pages uploaded")
    return len(images)


if __name__ == "__main__":
    client = get_qdrant_client()
    setup_collection(client)
    print("Qdrant connection OK ")