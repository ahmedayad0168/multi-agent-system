from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, asdict
from typing import List

from ragas.metrics import Faithfulness
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CriticalAgent")


@dataclass
class Critique:
    approved: bool
    faithfulness_score: float
    feedback: str
    warnings: List[str]


class CriticalAgent:
    def __init__(self, model_name: str = "llama3"):
        self.llm = LangchainLLMWrapper(ChatOllama(model=model_name))
        self.faithfulness = Faithfulness(llm=self.llm)

    def structural_check(self, answer: str) -> List[str]:
        issues = []
        if len(answer.split()) < 5:
            issues.append("Too short")
        if "[" not in answer or "]" not in answer: 
            issues.append("Missing citations")
        return issues

    async def evaluate(self, question: str, answer: str, context: List[str]) -> Critique:
        warnings = self.structural_check(answer)

        sample = SingleTurnSample(user_input= question, response= answer, retrieved_contexts= context)

        score = await self.faithfulness.single_turn_ascore(sample)
        score = round(score, 2)

        approved = score >= 0.8 and not warnings

        return Critique(approved= approved, faithfulness_score= score, feedback="OK" if approved else "Failed", warnings= warnings)


async def agentic_rag_loop(question: str, context: List[str], max_retries: int = 2):
    generator = ChatOllama(model="llama3")
    critic = CriticalAgent()

    feedback = "Initial"
    for _ in range(max_retries):
        prompt = f"""
        Context:
        {context}

        Question:
        {question}

        Feedback:
        {feedback}

        Rules:
        - Use ONLY context
        - Use citations [Dx]
        """

        response = await generator.ainvoke(prompt)
        answer = response.content

        result = await critic.evaluate(question, answer, context)

        if result.approved:
            return answer, result

        feedback = f"score={result.faithfulness_score}, issues={result.warnings}"

    return answer, result

# ============= test =================
# if __name__ == "__main__":
#     context_data = [
#         "Rags are pieces of old, torn, or worn-out fabric, often repurposed from clothing, towels, or sheets that are no longer usable for their original purpose.",
#         "They are commonly used for cleaning, dusting, wiping spills, and polishing due to their high absorbency and soft texture.",
#         "In industrial settings, rags are frequently employed as wipers in workshops, garages, and factories for cleaning machinery and tools.",
#         "Historically, rags were essential in papermaking, as cotton and linen fibers from shredded rags were used to produce high-quality paper.",
#         "Rags can also be recycled into new products like insulation, stuffing, or shoddy fabric, reducing textile waste and promoting sustainability."
#     ]
#     query = "What is Ragas and what does it do ?"
    
#     final_answer, final_critique = asyncio.run(agentic_rag_loop(query, context_data))
#     print(f"\n--- FINAL ANSWER ---\n{final_answer}")
#     print(f"\n--- CRITIQUE METRICS ---\n{asdict(final_critique)}")