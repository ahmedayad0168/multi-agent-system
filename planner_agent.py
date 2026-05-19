"""
planner_agent.py
────────────────────────────────────────────────────────────────
Multi-Agent Orchestrator for the Autonomous Research System.

Coordinates:
  • RAG Agent        → local document retrieval  (RAG.py)
  • Search Agent     → live web search           (search_agent.py)
  • Critic Agent     → hallucination detection   (critical_agent.py)
  • Memory Agent     → long-term context         (memory_agent.py)
  • Report Agent     → final synthesis           (memory_agent.final_report)

Flow
────
  User Goal
      │
      ▼
  PlannerAgent.create_plan()       ← LLM generates a JSON step list
      │
      ▼
  PlannerAgent.execute_plan()      ← topological execution, parallelises
      │                               independent steps automatically
      ├─► _run_rag_step()
      ├─► _run_search_step()       (runs in parallel with RAG)
      ├─► _run_critic_step()       (waits for RAG/Search)
      └─► _run_report_step()       (waits for Critic)
      │
      ▼
  ResearchReport                   ← structured final output
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import ollama

# ── Local agent imports ──────────────────────────────────────────────────────
# Each import is guarded; the planner degrades gracefully when a module's
# heavy dependencies (FAISS index, API keys) are not available at start-up.

try:
    from RAG import ask as _rag_ask
    from RAG import build_context as _rag_build_context
    from RAG import retrieve as _rag_retrieve
    _RAG_AVAILABLE = True
except Exception as _e:
    _RAG_AVAILABLE = False
    logging.getLogger("PlannerAgent").warning(
        f"RAG module unavailable (run database.py first?): {_e}"
    )

try:
    from search_agent import (
        build_context as _web_build_context,
        duckduck_search,
        merge_results,
        tavily_search,
        web_answer as _web_answer,
    )
    _SEARCH_AVAILABLE = True
except Exception as _e:
    _SEARCH_AVAILABLE = False
    logging.getLogger("PlannerAgent").warning(f"Search module unavailable: {_e}")

try:
    from critical_agent import CriticalAgent, agentic_rag_loop
    _CRITIC_AVAILABLE = True
except Exception as _e:
    _CRITIC_AVAILABLE = False
    logging.getLogger("PlannerAgent").warning(f"Critic module unavailable: {_e}")

try:
    from memory_agent import final_report as _final_report
    from memory_agent import remember, retrieve_memory
    _MEMORY_AVAILABLE = True
except Exception as _e:
    _MEMORY_AVAILABLE = False
    logging.getLogger("PlannerAgent").warning(f"Memory module unavailable: {_e}")


# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-14s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PlannerAgent")


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

class AgentType(str, Enum):
    """Canonical agent identifiers used in the execution plan."""
    RAG    = "rag"
    SEARCH = "search"
    CRITIC = "critic"
    MEMORY = "memory"
    REPORT = "report"


@dataclass
class PlanStep:
    """One step in the execution plan produced by the LLM."""
    step_id:    int
    agent:      AgentType
    task:       str
    depends_on: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "PlanStep":
        return cls(
            step_id    = int(d["step_id"]),
            agent      = AgentType(d["agent"]),
            task       = d["task"],
            depends_on = [int(x) for x in d.get("depends_on", [])],
        )


@dataclass
class AgentResult:
    """Output produced by a single agent step."""
    step_id:    int
    agent:      AgentType
    task:       str
    answer:     str
    approved:   Optional[bool]  = None
    score:      Optional[float] = None
    sources:    List[dict]      = field(default_factory=list)
    error:      Optional[str]   = None
    duration_s: float           = 0.0


@dataclass
class ExecutionPlan:
    goal:      str
    steps:     List[PlanStep]
    reasoning: str


@dataclass
class ResearchReport:
    goal:           str
    plan:           ExecutionPlan
    agent_results:  List[AgentResult]
    final_answer:   str
    critique_score: Optional[float]
    critique_passed: Optional[bool]
    total_duration_s: float


# ─────────────────────────────────────────────────────────────────────────────
# PlannerAgent
# ─────────────────────────────────────────────────────────────────────────────

class PlannerAgent:
    """
    Orchestrates the full multi-agent research pipeline.

    Parameters
    ----------
    model : str
        Ollama model used for planning and generation (default: llama3).
    critic_model : str
        Model passed to CriticalAgent — use a stronger model here if available.
    max_critic_retries : int
        How many regeneration attempts the Critic will trigger before giving up.
    """

    # ── Planner system prompt ────────────────────────────────────────────────
    _PLANNER_SYSTEM = """
You are a research planning agent inside a multi-agent AI system.

AVAILABLE AGENTS
────────────────
"rag"    – searches the LOCAL document database (PDFs / TXTs ingested by the user).
           Best for: domain-specific docs, uploaded papers, internal knowledge bases.
"search" – searches the LIVE WEB via Tavily + DuckDuckGo.
           Best for: current events, public facts, anything not in local docs.
"critic" – validates answers for hallucinations, missing citations, and low faithfulness.
           Always include after any information-gathering step.
"memory" – retrieves relevant past Q&A sessions from long-term memory.
           Include when the question might benefit from prior research.
"report" – synthesises all agent outputs into a final cited report.
           Always the LAST step.

PLANNING RULES
──────────────
1. Return ONLY a JSON object — no markdown, no explanation, no backticks.
2. Always include a "report" step as the final step.
3. Always include at least one "critic" step after information gathering.
4. Use "depends_on" to express ordering; steps without dependencies run in parallel.
5. If the query is purely about past sessions, skip rag/search and use memory + report.
6. If local documents are unlikely to have the answer, skip rag and use search only.

OUTPUT FORMAT (strict JSON):
{
  "reasoning": "<why you chose these agents and this order>",
  "steps": [
    {"step_id": 1, "agent": "rag",    "task": "<specific sub-task>", "depends_on": []},
    {"step_id": 2, "agent": "search", "task": "<specific sub-task>", "depends_on": []},
    {"step_id": 3, "agent": "critic", "task": "Validate gathered information",  "depends_on": [1, 2]},
    {"step_id": 4, "agent": "report", "task": "Synthesise and cite final answer","depends_on": [3]}
  ]
}
"""

    def __init__(
        self,
        model:               str = "llama3",
        critic_model:        str = "llama3",
        max_critic_retries:  int = 2,
    ):
        self.model              = model
        self.critic_model       = critic_model
        self.max_critic_retries = max_critic_retries

        if _CRITIC_AVAILABLE:
            self._critic = CriticalAgent(model_name=critic_model)
        else:
            self._critic = None

        logger.info(
            f"PlannerAgent ready │ model={model} │ critic_model={critic_model}"
        )

    # ── Plan creation ────────────────────────────────────────────────────────

    def create_plan(self, goal: str) -> ExecutionPlan:
        """
        Ask the LLM to produce a JSON execution plan for *goal*.
        Falls back to a sensible default if JSON parsing fails.
        """
        memory_hint = ""
        if _MEMORY_AVAILABLE:
            past = retrieve_memory(goal, k=2)
            if past:
                memory_hint = (
                    "\nRELEVANT PAST SESSIONS:\n"
                    + json.dumps(past, indent=2, ensure_ascii=False)
                )

        prompt = (
            self._PLANNER_SYSTEM
            + memory_hint
            + f"\n\nUSER GOAL:\n{goal}\n\nJSON PLAN:"
        )

        logger.info("Generating execution plan …")
        response = ollama.generate(model=self.model, prompt=prompt)
        raw = response["response"].strip()

        plan_dict = self._parse_json(raw)
        if plan_dict is None:
            logger.warning("Plan JSON parsing failed — using default plan.")
            plan_dict = self._default_plan_dict()

        steps = []
        for s in plan_dict.get("steps", []):
            try:
                steps.append(PlanStep.from_dict(s))
            except (KeyError, ValueError) as exc:
                logger.warning(f"Skipping malformed step {s}: {exc}")

        if not steps:
            steps = [PlanStep.from_dict(s) for s in self._default_plan_dict()["steps"]]

        return ExecutionPlan(
            goal      = goal,
            steps     = steps,
            reasoning = plan_dict.get("reasoning", "Default plan"),
        )

    # ── Plan execution ───────────────────────────────────────────────────────

    async def execute_plan(
        self, plan: ExecutionPlan
    ) -> Dict[int, AgentResult]:
        """
        Execute all steps respecting their dependency graph.
        Steps whose dependencies are already satisfied run concurrently.
        """
        results:   Dict[int, AgentResult] = {}
        completed: set[int]               = set()
        remaining                         = list(plan.steps)

        while remaining:
            # Steps whose dependencies are all done
            ready = [
                s for s in remaining
                if all(dep in completed for dep in s.depends_on)
            ]

            if not ready:
                failed = [s.step_id for s in remaining]
                logger.error(
                    f"Dependency deadlock — cannot execute steps {failed}. "
                    "Check for circular dependencies in the plan."
                )
                break

            logger.info(
                f"Running {len(ready)} step(s) in parallel: "
                + ", ".join(f"[{s.agent}]" for s in ready)
            )

            tasks = [self._dispatch(step, plan.goal, results) for step in ready]
            done  = await asyncio.gather(*tasks, return_exceptions=True)

            for step, outcome in zip(ready, done):
                if isinstance(outcome, Exception):
                    logger.error(f"Step {step.step_id} raised: {outcome}")
                    results[step.step_id] = AgentResult(
                        step_id = step.step_id,
                        agent   = step.agent,
                        task    = step.task,
                        answer  = "",
                        error   = str(outcome),
                    )
                else:
                    results[step.step_id] = outcome

                completed.add(step.step_id)
                remaining.remove(step)

        return results

    # ── Main entry point ─────────────────────────────────────────────────────

    async def run(self, goal: str) -> ResearchReport:
        """End-to-end orchestration: plan → execute → report."""
        t0 = time.perf_counter()

        logger.info(f"\n{'═'*60}\nNew research goal: {goal}\n{'═'*60}")

        plan = self.create_plan(goal)
        _print_plan(plan)

        results = await self.execute_plan(plan)

        # ── Extract report & critique from results ───────────────────────────
        report_res  = _find_agent(results, AgentType.REPORT)
        critic_res  = _find_agent(results, AgentType.CRITIC)

        final_answer    = report_res.answer  if report_res  else "⚠ No report generated."
        critique_score  = critic_res.score   if critic_res  else None
        critique_passed = critic_res.approved if critic_res else None

        return ResearchReport(
            goal             = goal,
            plan             = plan,
            agent_results    = list(results.values()),
            final_answer     = final_answer,
            critique_score   = critique_score,
            critique_passed  = critique_passed,
            total_duration_s = round(time.perf_counter() - t0, 2),
        )

    # ── Step dispatcher ──────────────────────────────────────────────────────

    async def _dispatch(
        self,
        step:    PlanStep,
        goal:    str,
        results: Dict[int, AgentResult],
    ) -> AgentResult:
        """Route a step to the correct handler."""
        t0 = time.perf_counter()
        logger.info(f"▶ Step {step.step_id} [{step.agent.upper()}] — {step.task}")

        loop = asyncio.get_event_loop()

        try:
            if step.agent == AgentType.RAG:
                result = await loop.run_in_executor(None, self._run_rag, goal)

            elif step.agent == AgentType.SEARCH:
                result = await loop.run_in_executor(None, self._run_search, goal)

            elif step.agent == AgentType.CRITIC:
                # Critic is already async-native
                result = await self._run_critic(goal, results)

            elif step.agent == AgentType.MEMORY:
                result = await loop.run_in_executor(None, self._run_memory, goal)

            elif step.agent == AgentType.REPORT:
                result = await loop.run_in_executor(
                    None, self._run_report, goal, results
                )

            else:
                raise NotImplementedError(f"Unknown agent: {step.agent}")

        except Exception as exc:
            logger.error(f"Step {step.step_id} [{step.agent}] failed: {exc}")
            result = AgentResult(
                step_id = step.step_id,
                agent   = step.agent,
                task    = step.task,
                answer  = "",
                error   = str(exc),
            )

        result.step_id    = step.step_id
        result.agent      = step.agent
        result.task       = step.task
        result.duration_s = round(time.perf_counter() - t0, 2)

        status = "✓" if not result.error else "✗"
        logger.info(
            f"{status} Step {step.step_id} [{step.agent.upper()}] "
            f"done in {result.duration_s}s"
        )
        return result

    # ── Individual agent runners ─────────────────────────────────────────────

    def _run_rag(self, goal: str) -> AgentResult:
        """Query local FAISS document index."""
        if not _RAG_AVAILABLE:
            return AgentResult(
                step_id=0, agent=AgentType.RAG, task="",
                answer="", error="RAG unavailable — run database.py first.",
            )

        answer, sources = _rag_ask(goal)
        return AgentResult(
            step_id=0, agent=AgentType.RAG, task="",
            answer=answer, sources=sources,
        )

    def _run_search(self, goal: str) -> AgentResult:
        """Live web search via Tavily + DuckDuckGo."""
        if not _SEARCH_AVAILABLE:
            return AgentResult(
                step_id=0, agent=AgentType.SEARCH, task="",
                answer="", error="Search unavailable — check API keys.",
            )

        answer = _web_answer(goal)
        return AgentResult(
            step_id=0, agent=AgentType.SEARCH, task="",
            answer=answer,
        )

    async def _run_critic(
        self,
        goal:    str,
        results: Dict[int, AgentResult],
    ) -> AgentResult:
        """
        Evaluate gathered answers for hallucinations.
        Prefers the RAG answer (has grounding in local docs).
        Falls back to the search answer if RAG is absent.
        """
        if not _CRITIC_AVAILABLE:
            return AgentResult(
                step_id=0, agent=AgentType.CRITIC, task="",
                answer="", error="Critic unavailable.",
            )

        rag_res    = _find_agent(results, AgentType.RAG)
        search_res = _find_agent(results, AgentType.SEARCH)

        if rag_res and not rag_res.error:
            # Pull raw text chunks for Ragas faithfulness check
            raw_chunks   = _rag_retrieve(goal)
            context_txts = [r["text"] for r in raw_chunks]

            answer, critique = await agentic_rag_loop(
                goal, context_txts, self.max_critic_retries
            )
            return AgentResult(
                step_id  = 0,
                agent    = AgentType.CRITIC,
                task     = "",
                answer   = answer,
                approved = critique.approved,
                score    = critique.faithfulness_score,
            )

        elif search_res and not search_res.error:
            # No local docs — do a lightweight structural check only
            warnings = self._critic.structural_check(search_res.answer)
            approved = len(warnings) == 0
            return AgentResult(
                step_id  = 0,
                agent    = AgentType.CRITIC,
                task     = "",
                answer   = search_res.answer,
                approved = approved,
                score    = None,
            )

        else:
            return AgentResult(
                step_id = 0, agent=AgentType.CRITIC, task="",
                answer  = "", error="No prior answers to critique.",
            )

    def _run_memory(self, goal: str) -> AgentResult:
        """Retrieve relevant past Q&A from long-term memory."""
        if not _MEMORY_AVAILABLE:
            return AgentResult(
                step_id=0, agent=AgentType.MEMORY, task="",
                answer="", error="Memory module unavailable.",
            )

        memories = retrieve_memory(goal)
        text = (
            json.dumps(memories, indent=2, ensure_ascii=False)
            if memories
            else "No relevant past sessions found."
        )
        return AgentResult(
            step_id=0, agent=AgentType.MEMORY, task="", answer=text
        )

    def _run_report(
        self,
        goal:    str,
        results: Dict[int, AgentResult],
    ) -> AgentResult:
        """
        Synthesise all agent outputs into one cited final report
        via memory_agent.final_report().
        """
        rag_res    = _find_agent(results, AgentType.RAG)
        search_res = _find_agent(results, AgentType.SEARCH)
        critic_res = _find_agent(results, AgentType.CRITIC)

        # Build combined evidence context
        context_parts: List[str] = []

        if rag_res and rag_res.answer:
            raw = _rag_retrieve(goal) if _RAG_AVAILABLE else []
            context_parts.append(_rag_build_context(raw) if raw else rag_res.answer)

        if search_res and search_res.answer:
            # Rebuild formatted web context
            t = tavily_search(goal) if _SEARCH_AVAILABLE else []
            d = duckduck_search(goal) if _SEARCH_AVAILABLE else []
            merged = merge_results(t, d) if _SEARCH_AVAILABLE else []
            context_parts.append(
                _web_build_context(merged) if merged else search_res.answer
            )

        combined_context = "\n\n".join(context_parts) or "No evidence context available."

        # The preferred answer to report is the critic's validated answer
        agent_answers: List[Dict[str, Any]] = []
        for r in results.values():
            if r.agent not in (AgentType.REPORT, AgentType.MEMORY) and r.answer:
                agent_answers.append({"agent": r.agent.value, "answer": r.answer})

        if not _MEMORY_AVAILABLE:
            # Fallback: ask the LLM directly to synthesise
            final = self._synthesise_fallback(goal, combined_context, agent_answers)
        else:
            final = _final_report(goal, combined_context, agent_answers)

        return AgentResult(
            step_id  = 0,
            agent    = AgentType.REPORT,
            task     = "",
            answer   = final,
            approved = critic_res.approved if critic_res else None,
            score    = critic_res.score    if critic_res else None,
        )

    def _synthesise_fallback(
        self,
        goal:          str,
        context:       str,
        agent_answers: List[Dict[str, Any]],
    ) -> str:
        """Fallback synthesiser when memory_agent is not available."""
        prompt = f"""
You are a research synthesis agent.

RULES:
- Use ONLY the provided evidence.
- Prefer answers with citations [D1], [W1] etc.
- Remove contradictions; flag conflicting sources.
- Be concise and accurate.

QUESTION: {goal}

EVIDENCE CONTEXT:
{context}

AGENT ANSWERS:
{json.dumps(agent_answers, indent=2)}

FINAL SYNTHESISED ANSWER:
"""
        response = ollama.generate(model=self.model, prompt=prompt)
        return response["response"]

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        """Strip markdown fences and parse JSON robustly."""
        # Remove ```json ... ``` wrappers
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                stripped = part.strip()
                if stripped.startswith("json"):
                    stripped = stripped[4:].strip()
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    continue
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _default_plan_dict() -> dict:
        return {
            "reasoning": "Comprehensive default plan: local docs + web + critique + report.",
            "steps": [
                {"step_id": 1, "agent": "rag",    "task": "Retrieve relevant local documents",          "depends_on": []},
                {"step_id": 2, "agent": "search",  "task": "Search the web for supporting information",  "depends_on": []},
                {"step_id": 3, "agent": "critic",  "task": "Validate and fact-check gathered answers",   "depends_on": [1, 2]},
                {"step_id": 4, "agent": "report",  "task": "Synthesise all evidence into final report",  "depends_on": [3]},
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_agent(
    results: Dict[int, AgentResult], agent: AgentType
) -> Optional[AgentResult]:
    return next((r for r in results.values() if r.agent == agent), None)


def _print_plan(plan: ExecutionPlan) -> None:
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  EXECUTION PLAN")
    print(f"  Goal: {plan.goal}")
    print(bar)
    print(f"  Reasoning: {plan.reasoning}\n")
    for step in plan.steps:
        deps = f"  (after: {step.depends_on})" if step.depends_on else ""
        print(f"  Step {step.step_id:>2}  [{step.agent.upper():<7}]  {step.task}{deps}")
    print(f"{bar}\n")


def print_report(report: ResearchReport) -> None:
    bar   = "═" * 60
    thin  = "─" * 60

    print(f"\n{bar}")
    print("  RESEARCH COMPLETE")
    print(f"{bar}")
    print(f"  Goal:            {report.goal}")
    print(f"  Total time:      {report.total_duration_s}s")

    if report.critique_score is not None:
        score_bar = "█" * int(report.critique_score * 10) + "░" * (10 - int(report.critique_score * 10))
        print(f"  Faithfulness:    {score_bar}  {report.critique_score:.2f}")

    passed = (
        "✓ PASSED" if report.critique_passed
        else ("✗ FAILED" if report.critique_passed is False else "─ N/A")
    )
    print(f"  Critique:        {passed}")

    print(f"\n{thin}")
    print("  AGENT RESULTS SUMMARY")
    print(thin)
    for r in sorted(report.agent_results, key=lambda x: x.step_id):
        status = "✗ ERROR" if r.error else "✓"
        print(f"\n  [{r.agent.upper():<7}] Step {r.step_id}  {status}  ({r.duration_s}s)")
        print(f"  Task: {r.task}")
        if r.error:
            print(f"  Error: {r.error}")
        elif r.answer:
            preview = r.answer[:250].replace("\n", " ")
            if len(r.answer) > 250:
                preview += " …"
            print(f"  Preview: {preview}")
        if r.score is not None:
            print(f"  Faithfulness score: {r.score}")

    print(f"\n{bar}")
    print("  FINAL REPORT")
    print(f"{bar}")
    print(report.final_answer)
    print(f"{bar}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI
# ─────────────────────────────────────────────────────────────────────────────

async def interactive_loop(planner: PlannerAgent) -> None:
    print("\n" + "═" * 60)
    print("  Multi-Agent Research System — type 'q' to quit")
    print("═" * 60)

    while True:
        try:
            goal = input("\nResearch Goal: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not goal:
            continue
        if goal.lower() in {"q", "quit", "exit"}:
            print("Goodbye!")
            break

        try:
            report = await planner.run(goal)
            print_report(report)

            # Persist the final answer to memory for future sessions
            if _MEMORY_AVAILABLE and report.final_answer:
                remember(question=goal, answer=report.final_answer)
                logger.info("Answer saved to long-term memory.")

        except Exception as exc:
            logger.error(f"Research pipeline failed: {exc}", exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Research System")
    parser.add_argument(
        "--model",
        default="llama3",
        help="Ollama model for planning and generation (default: llama3)",
    )
    parser.add_argument(
        "--critic-model",
        default="llama3",
        help="Model for the Critic agent — use a stronger model here (default: llama3)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Max critic-triggered regeneration attempts (default: 2)",
    )
    parser.add_argument(
        "--goal",
        default=None,
        help="Run a single research goal non-interactively and exit",
    )
    args = parser.parse_args()

    planner = PlannerAgent(
        model              = args.model,
        critic_model       = args.critic_model,
        max_critic_retries = args.retries,
    )

    if args.goal:
        # Single-shot mode
        async def _single() -> None:
            report = await planner.run(args.goal)
            print_report(report)
            if _MEMORY_AVAILABLE and report.final_answer:
                remember(question=args.goal, answer=report.final_answer)

        asyncio.run(_single())
    else:
        asyncio.run(interactive_loop(planner))
