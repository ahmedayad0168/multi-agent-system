from pathlib import Path
import json
import logging
import re

import faiss
import fitz
import numpy as np

from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "documents"
SAVE_DIR = BASE_DIR / "data"

INDEX_FILE = SAVE_DIR / "faiss.index"
CHUNKS_FILE = SAVE_DIR / "chunks.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 500
OVERLAP = 50
BATCH_SIZE = 32

SAVE_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO)
model = SentenceTransformer(MODEL_NAME)


def clean(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text):
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - OVERLAP
    return chunks


def load_files():

    all_chunks = []

    for path in DATA_DIR.iterdir():

        try:
            if path.suffix == ".txt":
                text = path.read_text(encoding= "utf-8")
                text = clean(text)
                chunks = chunk_text(text)
                for chunk in chunks:
                    all_chunks.append({"text": chunk, "source": path.name, "page": None})

            elif path.suffix == ".pdf":
                pdf = fitz.open(path)
                for page_num, page in enumerate(pdf):
                    text = clean(page.get_text())

                    if not text:
                        continue

                    chunks = chunk_text(text)
                    for chunk in chunks:
                        all_chunks.append({"text": chunk, "source": path.name, "page": page_num + 1})
                pdf.close()

            logging.info(f"Loaded: {path.name}")
        except Exception as e:
            logging.error(f"{path.name}: {e}")

    return all_chunks


def create_embeddings(chunks):
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, batch_size= BATCH_SIZE, normalize_embeddings= True, show_progress_bar= True)

    return np.asarray(vectors, dtype="float32")


def build_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save(index, chunks):
    faiss.write_index(index, str(INDEX_FILE))
    with open(CHUNKS_FILE, "w", encoding= "utf-8") as f:
        json.dump(chunks, f, ensure_ascii= False)


def ingest():
    chunks = load_files()
    if not chunks:
        print("No documents found.")
        return

    vectors = create_embeddings(chunks)
    index = build_index(vectors)
    save(index, chunks)
    print(f"\nDone. Total chunks: {len(chunks)}")


if __name__ == "__main__":

    ingest()