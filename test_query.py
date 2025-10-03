from sentence_transformers import SentenceTransformer
import chromadb


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="chroma_db")

def get_question_answer(question):
    query = question
    query_embedding = model.encode([query]).tolist()

    collection = client.get_collection(name="research_papers")

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )

    return {
        "documents": results["documents"],
        "metadata": results["metadatas"]
    }


