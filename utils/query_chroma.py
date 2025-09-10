# pip install chromadb sentence-transformers
import sys
import chromadb
from sentence_transformers import SentenceTransformer

def query_chroma(query, db_path="./chroma_db", collection_name="sgk_collection", top_k=5):
    # Load DB
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(collection_name)

    # Embedding query
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    q_embed = model.encode([f"query: {query}"]).tolist()

    # Search
    results = collection.query(query_embeddings=q_embed, n_results=top_k)

    print(f"\n🔎 Query: {query}\n")
    for i in range(len(results["ids"][0])):
        score = results["distances"][0][i]
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        #preview = doc[:200].replace("\n", " ")
        preview = doc.replace("\n", " ")
        print(f"[{i+1}] score={score:.3f}, meta={meta}\n{preview}...\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_chroma.py \"<your question>\" [db_path] [collection_name]")
        sys.exit(1)

    q = sys.argv[1]
    db_dir = sys.argv[2] if len(sys.argv) > 2 else "./chroma_db"
    coll = sys.argv[3] if len(sys.argv) > 3 else "sgk_collection"
    query_chroma(q, db_dir, coll)

