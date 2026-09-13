"""Deterministic mocks for the end-to-end pipeline benchmark.

FakeLLM implements the LLMClient surface the workflow touches
(generate_json + settings) and scripts stage-appropriate responses by
inspecting the prompt. FakeSearch returns realistic result payloads with
content, per query. Together they exercise the FULL production graph —
planner waves, search dedup, summarizer extraction, verification,
contradictions, confidence v2, budget accounting and stopping — with zero
network and zero nondeterminism.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Type

from app.core.config import Settings

PAGE_TEXTS = {
    "definition": (
        "Retrieval augmented generation is a technique that combines external "
        "search with language models. RAG systems retrieve documents at query "
        "time and condition generation on them. Retrieval augmented generation "
        "was introduced to ground model outputs in retrieved evidence and "
        "reduce fabrication."
    ),
    "adoption": (
        "Enterprise adoption of retrieval augmented generation reached 62% of surveyed "
        "companies in 2024 according to industry surveys. Adoption nearly doubled year "
        "over year as vector databases matured. However, enterprise adoption of RAG "
        "stalled at 62% in 2024 according to the later analyst report."
    ),
    "evidence": (
        "Benchmarks show retrieval augmented generation reduces hallucination rates by "
        "30% relative to parametric baselines. Grounding accuracy improved from 61% to "
        "74% on the evaluation suite. The 30% hallucination reduction was measured with "
        "gpt-class models on the KILT benchmark."
    ),
    "criticism": (
        "Retrieval augmented generation does not eliminate hallucination. Retrieval "
        "quality dominates end-to-end accuracy. Critics note evaluation setups often "
        "overstate RAG gains because baselines are untuned."
    ),
}

QUERIES_BY_STAGE = {
    "definition": "what is retrieval augmented generation",
    "adoption": "how widely is rag adopted in industry",
    "evidence": "what evidence supports rag reducing hallucinations",
    "criticism": "what are the criticisms of rag",
}


class FakeSearch:
    """SearchClient-compatible stub: deterministic results per query."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.calls: List[str] = []

    async def run_search(self, sub_questions) -> List[Dict[str, Any]]:
        """Deterministic results per query. Distinct URL per (stage, query) so
        the pass-level URL dedup never starves later contracts of evidence —
        real providers return different URLs for different queries."""
        results: List[Dict[str, Any]] = []
        seen_urls = set()
        for q in sub_questions or []:
            text = q[0] if isinstance(q, tuple) else str(q)
            self.calls.append(text)
            lowered = text.lower()
            slug = re.sub(r"[^a-z0-9]+", "-", lowered)[:40].strip("-") or "q"
            matched = False
            for stage, needle in QUERIES_BY_STAGE.items():
                if any(w in lowered for w in needle.split()[:3]):
                    matched = True
                    for i in range(3):
                        url = f"https://{stage}source{i}.org/{slug}"
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)
                        results.append({
                            "url": url,
                            "title": f"{stage} article {i}",
                            "snippet": PAGE_TEXTS[stage][:160],
                            "content": PAGE_TEXTS[stage],
                            "sub_question": text,
                            "published_at": "2024-06-15",
                        })
            if not matched:
                url = f"https://generalsource.org/{slug}"
                if url not in seen_urls:
                    seen_urls.add(url)
                    results.append({
                        "url": url,
                        "title": "General article",
                        "snippet": PAGE_TEXTS["definition"][:160],
                        "content": PAGE_TEXTS["definition"],
                        "sub_question": text,
                        "published_at": "2024-05-02",
                    })
        return results


class FakeLLM:
    """LLMClient-compatible stub with scripted stage responses."""

    def __init__(self, settings: Settings, critic_pass_on_iteration: int = 2):
        self.settings = settings
        self.critic_pass_on_iteration = critic_pass_on_iteration
        self.calls: List[Dict[str, str]] = []
        self.critic_iterations = 0

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        retries: int = 3,
        response_model: Type | None = None,
    ) -> Dict[str, Any]:
        stage = self._detect_stage(system_prompt, user_prompt)
        self.calls.append({"stage": stage, "user": user_prompt[:120]})
        self._record_usage(stage, system_prompt, user_prompt)
        if stage == "planner":
            return self._plan()
        if stage == "summarizer":
            return self._facts(user_prompt)
        if stage == "critic":
            self.critic_iterations += 1
            sufficient = self.critic_iterations >= self.critic_pass_on_iteration
            return {
                "is_sufficient": sufficient,
                "reason": "evidence covers the planned angles" if sufficient
                          else "coverage gaps remain",
                "improved_queries": [] if sufficient else [
                    "rag adoption methodology differences between surveys",
                ],
            }
        if stage == "synthesizer":
            return self._answer()
        return {}

    @staticmethod
    def _record_usage(stage: str, system_prompt: str, user_prompt: str) -> None:
        """Feed the run ledger like the real client does, so the e2e
        benchmark exercises budget accounting end-to-end."""
        try:
            from app.agents.budget import estimate_tokens
            from app.core.usage import get_run_usage

            usage = get_run_usage()
            if usage is None:
                return
            usage.record_llm(
                stage or "llm",
                prompt=f"{system_prompt}\n{user_prompt}",
                completion="",
                input_tokens=estimate_tokens(system_prompt) + estimate_tokens(user_prompt),
                output_tokens=400,
                model="mock",
            )
        except Exception:
            pass

    # -- stage detection -----------------------------------------------------

    @staticmethod
    def _detect_stage(system_prompt: str, user_prompt: str) -> str:
        blob = f"{system_prompt}\n{user_prompt}".lower()
        # Order matters: the summarizer prompt may embed sub-question context,
        # so its distinctive extraction instruction is checked FIRST.
        if "extract high-quality claims" in blob:
            return "summarizer"
        if "delegation contract" in blob or "sub-question" in blob or "planner" in blob:
            return "planner"
        if "critic" in blob or "sufficien" in blob:
            return "critic"
        if "synthesize" in blob or "final answer" in blob or "report" in blob:
            return "synthesizer"
        return "unknown"

    # -- scripted payloads ---------------------------------------------------

    @staticmethod
    def _plan() -> Dict[str, Any]:
        return {
            "sub_questions": [
                {
                    "question": "what is retrieval augmented generation",
                    "axis": "definition",
                    "search_type": "general",
                    "priority": 1,
                    "minimum_sources": 1,
                    "wave": 0,
                },
                {
                    "question": "how widely is rag adopted in industry",
                    "axis": "evidence",
                    "search_type": "statistical",
                    "priority": 2,
                    "minimum_sources": 1,
                    "wave": 0,
                },
                {
                    "question": "what evidence supports rag reducing hallucinations",
                    "axis": "evidence",
                    "search_type": "academic",
                    "priority": 3,
                    "minimum_sources": 1,
                    "wave": 0,
                },
                {
                    "question": "what are the criticisms of rag",
                    "axis": "criticism",
                    "search_type": "general",
                    "priority": 4,
                    "minimum_sources": 1,
                    "wave": 1,
                    "depends_on": [1],
                },
            ]
        }

    @staticmethod
    def _facts(user_prompt: str) -> Dict[str, Any]:
        """Extract claims from the sources embedded in the prompt."""
        facts = []
        url_pattern = re.compile(r"https://[a-z0-9./-]+")
        for url in url_pattern.findall(user_prompt)[:12]:
            stage = "definition"
            for key in PAGE_TEXTS:
                if key in url:
                    stage = key
                    break
            text = PAGE_TEXTS[stage]
            sentences = [s.strip() for s in re.split(r"(?<=[.])\s+", text) if s.strip()]
            for sentence in sentences[:2]:
                facts.append({
                    "claim": sentence,
                    "source": url,
                    "confidence": 0.85,
                    "direct_quote": sentence[:60],
                })
        return {"facts": facts}

    @staticmethod
    def _answer() -> Dict[str, Any]:
        answer = (
            "Retrieval augmented generation combines external search with language "
            "models to ground outputs in retrieved evidence [1]. "
            "Benchmarks show retrieval augmented generation reduces hallucination rates "
            "by 30% relative to parametric baselines [2]. "
            "However, retrieval augmented generation does not eliminate hallucination "
            "entirely [3]."
            "\n\nSources:\n"
            "[1] definitionsource0.org — https://definitionsource0.org/article\n"
            "[2] evidencesource0.org — https://evidencesource0.org/article\n"
            "[3] criticismsource0.org — https://criticismSource0.org/article"
        )
        return {"answer": answer}
