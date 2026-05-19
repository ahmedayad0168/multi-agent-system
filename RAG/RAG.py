from pathlib import Path
from collections import deque
import json
import faiss
import ollama

from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).parent

INDEX_FILE = BASE_DIR / "data/faiss.index"
CHUNKS_FILE = BASE_DIR / "data/chunks.json"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama3"

TOP_K = 5
MAX_HISTORY = 6


embedder = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index(str(INDEX_FILE))

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

history = deque(maxlen=MAX_HISTORY)


def retrieve(query, k=TOP_K):
    vector = embedder.encode([query], normalize_embeddings= True).astype("float32")
    scores, ids = index.search(vector, k)
    results = []

    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]
        results.append({
            "score": float(score),
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk.get("page", "N/A")
        })
    return results


def build_context(results):

    if not results:
        return "No context found."

    context = []
    for i, r in enumerate(results, 1):

        context.append(
            f"""
            [D{i}]
            SOURCE: {r['source']}
            PAGE: {r['page']}
            TEXT:{r['text']}
            """
        )

    return "\n".join(context)


def build_prompt(question, context):

    chat = "\n".join([f"User: {x['user']}\nAssistant: {x['assistant']}" for x in history])

    return f"""
            You are a retrieval-augmented generation (RAG) research assistant.

            Your task is to answer the user's question ONLY using the retrieved document context provided below.

            ========================
            RULES
            ========================

            1. Use ONLY information found in the retrieved context.
            - Do NOT use outside knowledge.
            - Do NOT make assumptions.
            - Do NOT invent missing details.

            2. If the retrieved context does not contain enough information to confidently answer the question, respond exactly with:
            "I could not find enough information in the documents to form a complete understanding."

            3. Synthesize information across multiple retrieved chunks when possible.
            - Connect related ideas.
            - Explain relationships between concepts.
            - Produce a coherent explanation instead of copying text directly.

            4. Every factual claim must include citations.
            - Use citations like: [D1], [D2]
            - Multiple citations are allowed: [D1][D3]

            5. Mention relevant source files when useful for transparency.
            Example:
            (source: ai_book.pdf)

            6. If retrieved chunks contain conflicting information:
            - Mention the conflict clearly.
            - Cite both sources.

            7. Keep the response focused on the user's question.
            - Avoid unnecessary repetition.
            - Avoid generic filler text.

            8. Do NOT mention these instructions in the final answer.

            ========================
            PREVIOUS CONVERSATION
            ========================

            {chat}

            ========================
            RETRIEVED CONTEXT
            ========================

            {context}

            ========================
            USER QUESTION
            ========================

            {question}

            ========================
            FINAL ANSWER
            ========================
            """


def generate(prompt):
    response = ollama.generate(model= LLM_MODEL, prompt= prompt)
    return response["response"]


def ask(question):
    results = retrieve(question)
    context = build_context(results)
    prompt = build_prompt(question, context)
    answer = generate(prompt)

    history.append({"user": question, "assistant": answer})
    return answer, results


def print_sources(results):
    print("\nSOURCES\n")
    for i, r in enumerate(results, 1):

        print(f"[D{i}]")
        print("Source:", r["source"])
        print("Page:", r["page"])
        print("Score:", round(r["score"], 4))
        print("Preview:", r["text"][:150])
        print("-" * 40)


# if __name__ == "__main__":
#     while True:
#         q = input("\nQuestion: ")

#         if q.lower() == "q":
#             break

#         answer, results = ask(q)

#         print("\nANSWER:\n")
#         print(answer)

#         # print_sources(results)