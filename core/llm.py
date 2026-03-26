# core/llm.py

# os lets us read environment variables
import os

# load_dotenv reads our .env file
# so GROQ_API_KEY becomes available via os.getenv()
from dotenv import load_dotenv

# ChatGroq is LangChain's wrapper around Groq API
# it handles the API calls, retries, formatting for us
from langchain_groq import ChatGroq

# These are LangChain message types
# SystemMessage = instructions/rules for the LLM
# HumanMessage  = the actual user question
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables from .env file
# must call this before os.getenv() or it returns None
load_dotenv()

# Initialize Groq LLM
# model: llama3-70b-8192
#   70b = 70 billion parameters (very capable)
#   8192 = context window size (how much text it can see)
# temperature: controls randomness
#   0.2 = mostly deterministic, focused answers
#   0.0 = completely deterministic
#   1.0 = very creative/random
# groq_api_key: reads from .env file
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY")
)


def get_answer(query: str, context_list: list[dict]) -> dict:
    """
    Takes user question and retrieved context chunks
    Builds a prompt and sends to Groq
    Returns answer with sources

    Args:
        query        : the user's question string
        context_list : list of dicts from retriever.py
                       each dict has text, source, similarity

    Returns:
        dict with answer string and sources list
    """

    # Handle case where no context was retrieved
    # this happens if knowledge base is empty
    if not context_list:
        return {
            "answer": "I don't have enough information to answer that. Please add relevant content to the knowledge base first.",
            "sources": []
        }

    # Build the context string from retrieved chunks
    # we number each chunk so LLM can reference them
    # and we include the source file name
    context_text = ""
    for i, chunk in enumerate(context_list):
        context_text += f"\n[{i+1}] Source: {chunk['source']}\n"
        context_text += f"{chunk['text']}\n"
        # example output:
        # [1] Source: training.txt
        # Progressive overload is the gradual increase...
        # [2] Source: nutrition.txt
        # Protein intake should be 1.6 to 2.2 grams...

    # System prompt — this sets the rules for the LLM
    # this is the most important part of prompt engineering
    # we are telling the LLM exactly how to behave
    system_prompt = """You are a knowledgeable fitness and gym expert assistant.

Your job is to answer questions about fitness, training, nutrition and supplements.

STRICT RULES you must follow:
1. Answer ONLY using the context provided below
2. If the answer is not in the context say exactly: "I don't have specific information about that in my knowledge base"
3. Do not make up information or use outside knowledge
4. Keep answers clear and practical
5. Reference which source your answer came from

This ensures your answers are accurate and trustworthy."""

    # Human message — the actual question with context
    # we inject the retrieved chunks right into the message
    # the LLM sees question + relevant content together
    human_message = f"""Context information:
{context_text}

Question: {query}

Please answer based only on the context above."""

    # Send to Groq via LangChain
    # we pass a list of messages — this is chat format
    # SystemMessage sets behavior, HumanMessage is the query
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message)
    ])

    # response.content is the actual answer text string
    answer = response.content

    # Build sources list for citations
    # user can see exactly where the answer came from
    sources = []
    for chunk in context_list:
        source_entry = {
            "file": chunk["source"],
            "similarity": chunk["similarity"],
            "preview": chunk["text"][:100] + "..."
            # first 100 chars of chunk as preview
        }
        sources.append(source_entry)

    return {
        "answer": answer,
        "sources": sources
    }


# Test block — runs only with: python core/llm.py
if __name__ == "__main__":

    # We need to import retriever here to get real context
    # this tests the FULL pipeline: retrieve → answer
    import sys
    sys.path.append(".")
    from core.retriever import retrieve_context

    test_query = "what is progressive overload and how do I apply it?"

    print(f"\n💬 Question: {test_query}")
    print("=" * 60)
    print("🔍 Retrieving context...")

    # Get relevant chunks from ChromaDB
    context = retrieve_context(test_query, k=3)

    print(f"✅ Found {len(context)} relevant chunks")
    print("\n🤖 Asking Groq...")

    # Get answer from LLM
    result = get_answer(test_query, context)

    print(f"\n📝 Answer:\n{result['answer']}")
    print(f"\n📚 Sources:")
    for s in result['sources']:
        print(f"  → {s['file']} ({s['similarity']}% match)")
        print(f"     {s['preview']}")