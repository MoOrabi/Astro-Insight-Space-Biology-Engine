from sentence_transformers import SentenceTransformer
import chromadb

# 1. تحميل الموديل من HuggingFace (مجاني وسريع)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

query = "Mice behavior during a flight"
query_embedding = model.encode([query]).tolist()

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(name="research_papers")

results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

print("Query results:")
for doc in results["documents"][0]:
    print("-", doc)