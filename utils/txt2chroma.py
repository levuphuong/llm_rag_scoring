# pip install chromadb sentence-transformers langchain-community
import sys, re, unicodedata, os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_chroma(txt_path, db_path="./chroma_db", collection_name="sgk_collection"):
    # 1) Load text file và tách theo trang
    with open(txt_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Regex tách theo "--- Page X ---"
    pages = re.split(r"--- Page \d+ ---", raw)
    docs = [normalize_text(p) for p in pages if p.strip()]

    # 2) Chunk (900 ký tự, overlap 120)
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = []
    metadatas = []
    for i, doc in enumerate(docs, start=1):
        split = splitter.split_text(doc)
        for j, chunk in enumerate(split):
            chunks.append(chunk)
            metadatas.append({"page": i, "chunk": j})

    # 3) Embedding
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    passages = [f"passage: {c}" for c in chunks]
    embeddings = model.encode(passages, batch_size=32, show_progress_bar=True)

    # 4) Save vào ChromaDB
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(collection_name)

    for idx, text in enumerate(chunks):
        collection.add(
            ids=[f"{idx}"],
            documents=[text],
            embeddings=[embeddings[idx].tolist()],
            metadatas=[metadatas[idx]]
        )

    print(f"✅ Đã build DB từ {txt_path} → {db_path} (collection={collection_name}, n_chunks={len(chunks)})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python txt2chroma.py <input.txt> [db_path] [collection_name]")
        sys.exit(1)

    txt_file = sys.argv[1]
    db_dir = sys.argv[2] if len(sys.argv) > 2 else "./chroma_db"
    coll = sys.argv[3] if len(sys.argv) > 3 else "sgk_collection"
    build_chroma(txt_file, db_dir, coll)

