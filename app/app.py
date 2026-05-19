from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


from search_agent.search_agent import search_agent
from RAG.RAG import RAG
from critical_agent.critical_agent.critical_agent import agentic_rag_loop
from memory_agent.memory_agent import memory_agent


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("MultiAgentSystem")

CRITIC_THRESHOLD = 0.90


class MultiAgentSystem:
    def build_final_response(self, question, answer, score, source):
        return {
            "question": question,
            "answer": answer,
            "score": score,
            "source": source
        }

    def save_memory(self, data):
        try:
            memory_agent.save_memory(data)
        except Exception as e:
            logger.error(f"Memory save failed: {e}")

    async def local_pipeline(self, question: str, memory_context: str):
        logger.info("Searching local docs...")
        rag_results = RAG.retrieve(question)
        if not rag_results:
            return None

        context = [r["text"] for r in rag_results if "text" in r]
        logger.info("Running local agent loop...")
        answer, critique = await agentic_rag_loop(question= question, context= context)

        return {
            "answer": answer,
            "score": critique.faithfulness_score,
            "source": "local_rag"
        }

    async def web_pipeline(self, question: str, memory_context: str):
        logger.info("Searching web...")
        tavily_task = asyncio.to_thread(search_agent.tavily_search, question)
        duck_task = asyncio.to_thread(search_agent.duckduck_search, question)

        tavily, duck = await asyncio.gather(tavily_task, duck_task)
        merged = search_agent.merge_results(tavily, duck)
        if not merged:
            return None

        context = [r.get("content", "") for r in merged if r.get("content")]
        logger.info("Running web agent loop...")
        answer, critique = await agentic_rag_loop(question= question, context= context)

        return {
            "answer": answer,
            "score": critique.faithfulness_score,
            "source": "web_search"
        }

    async def run(self, question: str):
        logger.info(f"Question: {question}")
        try:
            memory_context = memory_agent.retrieve_memory(question)
        except:
            memory_context = ""

        local = await self.local_pipeline(question, memory_context)
        if local:
            logger.info(f"Local score: {local['score']}")
            if local["score"] >= CRITIC_THRESHOLD:
                result = self.build_final_response(question, local["answer"], local["score"], local["source"])

                self.save_memory(result)
                return result

        web = await self.web_pipeline(question, memory_context)
        if web:
            logger.info(f"Web score: {web['score']}")
            result = self.build_final_response(question, web["answer"], web["score"], web["source"])

            self.save_memory(result)
            return result

        return {
            "question": question,
            "answer": "No result found",
            "score": 0.0,
            "source": "none"
        }


async def main():
    system = MultiAgentSystem()

    while True:
        q = input("\nQuestion (q to quit): ")
        if q.lower() == "q":
            break

        result = await system.run(q)

        print("\n==== FINAL ====")
        print(result["answer"])
        print("Score:", result["score"])
        print("Source:", result["source"])


if __name__ == "__main__":
    asyncio.run(main())