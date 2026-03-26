# core/retriever.py

# Same embedding model as ingest.py
# IMPORTANT: must be the SAME model used during ingestion
# Different model = different vector space = wrong results
from sentence_transformers import SentenceTransformer

# ChromaDB to connect to our existing vector store
import chromadb

# Load the same embedding model
# This uses cached version — no re-download
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to the SAME vectorstore folder as ingest.py
# PersistentClient reads the data we already stored
client = chromadb.PersistentClient(path="./vectorstore")

# Get our existing collection
# NOTE: get_or_create_collection — if for some reason
# it doesn't exist yet, it creates it (prevents crash)
collection = client.get_or_create_collection(
    name="gym_knowledge",
    metadata={"hnsw:space": "cosine"}
)


def retrieve_context(query: str, k: int = 3) -> list[dict]:
    """
    Takes a user question (query)
    Finds top-k most similar chunks from ChromaDB
    Returns list of dicts with text and source
    
    Example:
    query = "what is progressive overload?"
    returns = [
        {"text": "Progressive overload means...", "source": "training.txt"},
        {"text": "To apply progressive overload...", "source": "training.txt"},
        {"text": "Compound exercises...", "source": "training.txt"},
    ]
    """

    # Check if collection has any data
    # If someone calls this before ingesting anything
    # we return empty list instead of crashing
    if collection.count() == 0:
        print("⚠️  No documents ingested yet. Run ingest first.")
        return []

    # Convert the user's question into a vector
    # Same process as when we embedded chunks during ingestion
    # query_embedding shape: (384,) — one vector for the question
    query_embedding = embedding_model.encode(query).tolist()

    # Search ChromaDB for similar vectors
    # query_embeddings = the question vector (in a list)
    # n_results = how many chunks to return (our k)
    # include = what data to return alongside the vectors
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        # min() prevents error if we have fewer chunks than k
        include=["documents", "metadatas", "distances"]
        # documents = original chunk text
        # metadatas = source filename info
        # distances = similarity scores (lower = more similar)
    )

    # results structure from ChromaDB:
    # {
    #   "documents": [["chunk1 text", "chunk2 text", "chunk3 text"]],
    #   "metadatas": [[{"source": "training.txt"}, ...]],
    #   "distances": [[0.12, 0.34, 0.67]]
    # }
    # Note the extra nesting — ChromaDB wraps in outer list
    # because it supports batch queries
    # We only sent one query so we take index [0]

    # Extract the actual lists from nested structure
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Build clean list of results
    # We combine text + source + similarity score
    context_list = []

    for doc, meta, dist in zip(documents, metadatas, distances):

        # Convert distance to similarity score
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # We convert to percentage: 1 - dist = similarity
        # e.g. distance 0.12 → similarity 0.88 → 88% similar
        similarity = round((1 - dist) * 100, 2)

        context_list.append({
            "text": doc,
            "source": meta["source"],
            "similarity": similarity
        })

    return context_list


# Test block — runs only when you run this file directly
# python core/retriever.py
if __name__ == "__main__":

    # Test query
    test_query = "what is progressive overload?"

    print(f"\n🔍 Query: {test_query}")
    print("=" * 50)

    results = retrieve_context(test_query, k=3)

    for i, result in enumerate(results):
        print(f"\n📄 Result {i+1}")
        print(f"Source:     {result['source']}")
        print(f"Similarity: {result['similarity']}%")
        print(f"Text:       {result['text'][:150]}...")
        # [:150] = first 150 characters only for readability