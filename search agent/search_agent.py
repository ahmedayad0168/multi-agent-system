import os
import ollama

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun


load_dotenv()
MODEL = os.getenv("OLLAMA_MODEL", "llama3")
MAX_RESULTS = 4


def tavily_search(query):
    try:
        tool = TavilySearchResults(max_results= MAX_RESULTS)
        results = tool.invoke(query)
        return results

    except Exception as e:
        print("Tavily Error:", e)
        return []


def duckduck_search(query):
    try:
        tool = DuckDuckGoSearchRun()
        result = tool.invoke(query)
        return [{
            "title": "DuckDuckGo Result",
            "content": str(result),
            "source": "duckduckgo"
        }]

    except Exception as e:
        print("DuckDuckGo Error:", e)
        return []


def merge_results(*results_lists):
    merged = []
    seen = set()

    for results in results_lists:
        for r in results:
            content = r.get("content", "").strip()
            if not content:
                continue

            key = content[:100]
            if key in seen:
                continue

            seen.add(key)
            merged.append(r)

    return merged


def build_context(results):
    if not results:
        return "No web evidence found."

    context = []
    for i, r in enumerate(results, 1):
        context.append(
            f"""
            [W{i}]
            TITLE: {r.get("title", "")}

            CONTENT:
            {r.get("content", "")}
            """
            )
    return "\n".join(context)


def web_answer(question):
    tavily_results = tavily_search(question)
    duck_results = duckduck_search(question)
    results = merge_results(tavily_results, duck_results)
    context = build_context(results)

    prompt = f"""
    You are a web research assistant.

    Rules:
    - Use ONLY provided web evidence
    - Do NOT hallucinate
    - Add citations like [W1], [W2]
    - If evidence is weak, say so

    QUESTION:
    {question}

    WEB EVIDENCE:
    {context}

    FINAL ANSWER:
    """

    response = ollama.generate(model= MODEL, prompt= prompt)
    return response["response"]


# if __name__ == "__main__":
#     question = input("Ask: ")
#     answer = web_answer(question)

#     print("\nFINAL ANSWER:\n")
#     print(answer)