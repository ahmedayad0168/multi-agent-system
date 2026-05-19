from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import ollama

from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).parent

MEMORY_DIR = BASE_DIR / "memory"
MEMORY_DIR.mkdir(exist_ok=True)

MEMORY_FILE = MEMORY_DIR / "memory.json"
MEMORY_INDEX = MEMORY_DIR / "memory.index"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama3"

TOP_K = 3

embedder = SentenceTransformer(EMBED_MODEL)


def load_memory():
    if not MEMORY_FILE.exists():
        return []

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent= 2, ensure_ascii= False)


def build_memory_index(memory):
    if not memory:
        return None

    texts = [f"{m['question']} {m['answer']}" for m in memory]
    vectors = embedder.encode(texts, normalize_embeddings= True)
    vectors = np.asarray(vectors, dtype= "float32")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(MEMORY_INDEX))

    return index


def load_memory_index():
    if not MEMORY_INDEX.exists():
        return None

    return faiss.read_index(str(MEMORY_INDEX))


def retrieve_memory(query, k=TOP_K):
    memory = load_memory()
    if not memory:
        return []

    index = load_memory_index()
    if index is None:
        index = build_memory_index(memory)

    vector = embedder.encode([query], normalize_embeddings= True)
    vector = np.asarray(vector, dtype= "float32")
    scores, ids = index.search(vector, k)

    results = []
    for idx in ids[0]:
        if idx == -1:
            continue

        results.append(memory[idx])

    return results


def remember(question, answer):
    memory = load_memory()

    item = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer
    }

    memory.append(item)
    save_memory(memory)
    build_memory_index(memory)


FINAL_PROMPT = """
You are the final synthesis agent in a multi-agent system.

Rules:
- Use ONLY the provided evidence context
- Prefer answers with citations
- Remove contradictions
- Ignore hallucinated answers
- Keep the final response concise, clear, and accurate
- Mention uncertainty if evidence is weak
"""


def final_report(question, context, agent_answers):
    memory = retrieve_memory(question)
    prompt = f"""
    {FINAL_PROMPT}

    QUESTION:
    {question}

    EVIDENCE CONTEXT:
    {context}

    RELEVANT MEMORY:
    {json.dumps(memory, indent=2, ensure_ascii=False)}

    AGENT ANSWERS:
    {json.dumps(agent_answers, indent=2, ensure_ascii=False)}

    FINAL ANSWER:
    """

    response = ollama.generate(model= LLM_MODEL, prompt= prompt)
    final_answer = response["response"]
    remember(question= question, answer= final_answer)

    return final_answer


# if __name__ == "__main__":
#     question = "What is Python used for?"

#     context = """
#     [D1] Python is widely used in AI and machine learning.

#     [D2] Python is also used in web development,
#     automation, and data science.
#     """

#     agent_answers = [

#         {
#             "agent": "agent_1",
#             "answer": "Python is only used for games."
#         },

#         {
#             "agent": "agent_2",
#             "answer": (
#                 "Python is used in AI, web development, "
#                 "automation, and machine learning. [D1][D2]"
#             )
#         }
#     ]

#     result = final_report(question, context, agent_answers)

#     print("\nFINAL ANSWER:\n")
#     print(result)