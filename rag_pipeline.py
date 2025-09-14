import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np

load_dotenv()

# --- Pinecone setup ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "atlan-docs"  # your index from dashboard
index = pc.Index(INDEX_NAME)
NAMESPACE = "__default__"

# --- Embeddings ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Groq client ---
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

def query_rag(user_query: str, top_k: int = 10):
    # --- Embed query ---
    query_vec = embed_model.encode([user_query])[0].tolist()
    print("🔹 Query embedding vector (first 10 dims):", query_vec[:10], "...")

    # --- Pinecone search ---
    results = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE,
        include_values=True
    )

    # --- Debug: show top 5 similarities manually ---
    if results["matches"]:
        similarities = []
        for match in results["matches"]:
            # Check if 'values' key exists and is not empty before proceeding
            if "values" in match and match["values"]:
                chunk_vec = np.array(match["values"])
                q_vec_np = np.array(query_vec)
                cos_sim = np.dot(q_vec_np, chunk_vec) / (np.linalg.norm(q_vec_np) * np.linalg.norm(chunk_vec))
                similarities.append((match["id"], cos_sim, match["metadata"].get("url", "N/A")))
            else:
                # Handle cases where values are missing to prevent crash
                similarities.append((match["id"], 0.0, match["metadata"].get("url", "N/A")))

        similarities.sort(key=lambda x: x[1], reverse=True)
        print("🔹 Top 5 most similar chunks:")
        for cid, score, url in similarities[:5]:
            print(f"ID: {cid}, Score: {score:.4f}, URL: {url}")
    else:
        print("⚠️ No matches found in Pinecone!")
        return "I don't know.", []

    # --- Extract text and sources ---
    contexts = [m["metadata"]["text"] for m in results["matches"]]
    sources = [m["metadata"].get("url", "N/A") for m in results["matches"]]

    # --- Prepare prompt for Groq ---
    context_str = "\n\n".join(contexts)
    prompt = f"""You are a helpful support agent.
Use the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context_str}

Question: {user_query}
Answer:"""

    # --- Groq completion ---
    completion = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
    )

    print("Matches returned from Pinecone:", len(results["matches"]))
    return completion.choices[0].message.content, list(set(sources))
