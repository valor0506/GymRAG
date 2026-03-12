import os 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import uuid

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./vector_store")
collection = client.get_or_create_collection(
    name="gym_knowledge",
    metadata={"hnsw:space": "cosine"}
)

def ingest_documents(folder_path:str)->str:
    splitter = RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=64,separators=["\n\n","\n","."," ",""])
    total_chunks = 0
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(folder_path,filename)
        with open(filepath,'r',encoding='utf-8') as f:
            text = f.read()
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        embeddings = embedding_model.encode(chunks)
        embeddings_list = embeddings.tolist()
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": filename} for _ in chunks]
        collection.add(ids=ids,embeddings=embeddings_list,metadatas=metadatas)
        total_chunks += len(chunks)
        print(f"Ingested {len(chunks)} chunks from {filename}")
    return f"Ingestion Complete. Total chunks ingested: {total_chunks}"



if __name__ == "__main__":
    result = ingest_documents("./knowledge")
    print(result)