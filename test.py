import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import numpy as np
import time

load_dotenv()

# --- Pinecone setup ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "atlan-docs"
index = pc.Index(INDEX_NAME)
NAMESPACE = "__default__"

# --- Embeddings ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def test_query_rag(user_query: str, top_k: int = 50):
    """
    Tests the RAG retrieval pipeline without calling the LLM.
    Prints the retrieved contexts and their similarity scores.
    """
    print(f"Testing RAG for query: '{user_query}'")
    
    # --- Embed query ---
    print("Embedding query...")
    query_vec = embed_model.encode([user_query])[0].tolist()
    print("🔹 Query embedding vector (first 10 dims):", query_vec[:10], "...")
    
    # --- Pinecone search ---
    print("Searching Pinecone for matches...")
    start_time = time.time()
    results = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE,
        include_values=True
    )
    end_time = time.time()
    print(f"Search completed in {end_time - start_time:.4f} seconds.")

    # --- Print retrieved results ---
    if results["matches"]:
        print(f"✅ Found {len(results['matches'])} matches in Pinecone.")
        
        # Manually calculate and print top similarities for debugging
        similarities = []
        for match in results["matches"]:
            chunk_vec = np.array(match["values"])
            q_vec_np = np.array(query_vec)
            # Use dot product for cosine similarity with normalized vectors
            cos_sim = np.dot(q_vec_np, chunk_vec) 
            similarities.append((match["id"], cos_sim, match["metadata"].get("url", "N/A"), match["metadata"].get("text", "No text provided")))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print("\n🔹 Top 5 most similar chunks:")
        for i, (cid, score, url, text) in enumerate(similarities[:5]):
            print(f"--- Match {i+1} ---")
            print(f"ID: {cid}, Score: {score:.4f}")
            print(f"URL: {url}")
            print(f"Content (first 150 chars): {text[:150].replace('\n', ' ')}...")
            print("-" * 20)
    else:
        print("⚠️ No matches found in Pinecone!")

    return results["matches"]

if __name__ == "__main__":
    # Example queries to test the RAG pipeline
    queries_to_test = [
        "how to integrate atlan with always on",
        "steps to automate data governance",
        "what are Atlan governance workflows",
        "what is active metadata in atlan",
        "how do i connect a snowflake database to atlan"
    ]
    
    for q in queries_to_test:
        test_query_rag(q)
        print("\n" + "="*50 + "\n")
