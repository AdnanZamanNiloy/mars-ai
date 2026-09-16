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


# ===========================================================================
# Golden benchmark mocks (bench/golden/)
# ===========================================================================
# The classes above are scripted to a single RAG query — running a 20-40
# query golden set through them yields one generic result repeated. These
# subclasses extend the SAME production-graph exercise with per-query
# deterministic fixtures (bench/golden/fixtures_v1.py), so every golden query
# flows through the real planner/search/summarizer/verifier/dedupe/grade/
# contradiction/confidence/synthesis pipeline with domain-appropriate
# evidence, zero network and zero nondeterminism.
#
# They REUSE FakeLLM's stage detection and payload-shape contract; only the
# data source changes (fixture pack selected by query id / routed topic).


def _golden_url_index() -> Dict[str, Dict[str, Any]]:
    from bench.golden import fixtures_v1

    index: Dict[str, Dict[str, Any]] = {}
    for pages in fixtures_v1.TOPIC_PACKS.values():
        for page in pages:
            index[page["url"]] = page
    return index


class GoldenFakeSearch(FakeSearch):
    """Deterministic search keyed by the golden query's topic pack.

    `query` (the golden query text) routes to a topic; each sub-question is
    routed independently so multi-angle queries return evidence from the
    axes they actually asked about. URLs are stable per topic+page."""

    def __init__(self, settings: Settings, query: str = ""):
        super().__init__(settings)
        self.query = query
        self.topic = self._route(query) or "generic"

    @staticmethod
    def _route(text: str) -> str:
        from bench.golden import fixtures_v1

        return fixtures_v1.route_topic(text)

    async def run_search(self, sub_questions) -> List[Dict[str, Any]]:
        from bench.golden import fixtures_v1

        results: List[Dict[str, Any]] = []
        seen_urls: set = set()
        queries = [q[0] if isinstance(q, tuple) else str(q) for q in (sub_questions or [])]
        for index, question in enumerate(queries):
            self.calls.append(question)
            topic = self._route(question)
            if topic == "generic":
                topic = self.topic
            pack = fixtures_v1.pages_for_topic(topic)
            # Rotate the pack per sub-question: each contract gets a DISTINCT
            # slice so the production per-contract search dedup does not starve
            # later contracts of evidence (which would collapse a multi-domain
            # query onto one publisher and understate coverage).
            if pack:
                offset = index % len(pack)
                rotated = pack[offset:] + pack[:offset]
            else:
                rotated = pack
            for page in rotated[: len(pack)]:
                if page["url"] in seen_urls:
                    continue
                seen_urls.add(page["url"])
                result = dict(page)
                result["sub_question"] = question
                results.append(result)
        return results


class GoldenFakeLLM(FakeLLM):
    """LLMClient-compatible stub serving the golden fixtures deterministically.

    Query id selects the topic; the planner emits one contract per expected
    research dimension, the summarizer extracts fixture page sentences as
    claims, and the synthesizer composes a sectioned, fully-cited report from
    the evidence block the production prompt hands it."""

    def __init__(self, settings: Settings, query_id: str = "", query: str = "",
                 critic_pass_on_iteration: int = 1):
        super().__init__(settings, critic_pass_on_iteration=critic_pass_on_iteration)
        self.query_id = query_id
        self.query = query
        from bench.golden import fixtures_v1

        self.topic = fixtures_v1.TOPIC_BY_QUERY_ID.get(query_id) or fixtures_v1.route_topic(query)

    @staticmethod
    def _detect_stage(system_prompt: str, user_prompt: str) -> str:
        """Golden stage detection: match the SYSTEM PROMPT signature only.

        The base mock scans system+user together, and the synthesizer's user
        prompt can embed evidence/context text that trips the planner branch
        (`sub-question`/`planner`), silently routing a synthesis call to the
        planner payload — which is exactly how a golden run produced the
        extractive fallback instead of a written report. The stable contract
        is the system prompt: its distinctive phrases belong to exactly one
        stage. Check the synthesizer's unique signature first, then delegate
        to the base heuristic for the rest."""
        system_low = (system_prompt or "").lower()
        if "final synthesis agent" in system_low:
            return "synthesizer"
        if "extract high-quality claims" in system_low:
            return "summarizer"
        return FakeLLM._detect_stage(system_prompt, user_prompt)

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
            return self._golden_plan()
        if stage == "summarizer":
            return self._golden_facts(user_prompt)
        if stage == "critic":
            self.critic_iterations += 1
            sufficient = self.critic_iterations >= self.critic_pass_on_iteration
            return {
                "is_sufficient": sufficient,
                "reason": "evidence covers the planned angles" if sufficient
                          else "coverage gaps remain",
                "improved_queries": [] if sufficient else ["additional evidence angle"],
            }
        if stage == "synthesizer":
            return self._golden_answer(user_prompt)
        return {}

    # -- planner -------------------------------------------------------------
    def _expected_dimensions(self) -> List[str]:
        from bench.golden import fixtures_v1

        query = fixtures_v1.expected_by_id().get(self.query_id)
        dims = list(query.get("required_dimensions", [])) if query else []
        return dims or ["definition", "evidence"]

    def _golden_plan(self) -> Dict[str, Any]:
        dims = self._expected_dimensions()
        sub_questions = []
        for i, axis in enumerate(dims):
            sub_questions.append({
                "question": self._question_for_axis(axis),
                "axis": axis,
                "search_type": self._search_type_for_axis(axis),
                "priority": i + 1,
                "minimum_sources": 1,
                "wave": 0,
            })
        return {"sub_questions": sub_questions}

    def _question_for_axis(self, axis: str) -> str:
        base = self.query or "the research question"
        templates = {
            "definition": f"what is {base} definition and meaning",
            "mechanism": f"how does {base} work mechanism",
            "evidence": f"what evidence supports {base}",
            "statistical": f"statistics and data on {base}",
            "trend": f"current trend in {base}",
            "comparison": f"compare {base} options and tradeoffs",
            "causal": f"what causes {base} causal drivers",
            "policy": f"policy options for {base}",
        }
        return templates.get(axis, f"{base} {axis}")

    @staticmethod
    def _search_type_for_axis(axis: str) -> str:
        return {
            "definition": "general",
            "mechanism": "general",
            "evidence": "academic",
            "statistical": "statistical",
            "trend": "statistical",
            "comparison": "comparison",
            "causal": "academic",
            "policy": "news",
        }.get(axis, "general")

    # -- summarizer ----------------------------------------------------------
    def _golden_facts(self, user_prompt: str) -> Dict[str, Any]:
        from bench.golden import fixtures_v1

        index = _golden_url_index()
        url_pattern = re.compile(r"https://[a-z0-9./_-]+", re.IGNORECASE)
        facts: List[Dict[str, Any]] = []
        seen: set = set()
        for url in url_pattern.findall(user_prompt):
            page = index.get(url.rstrip(".,;)"))
            if page is None or url in seen:
                continue
            seen.add(url)
            for sentence in fixtures_v1.claim_sentences(page.get("content", "")):
                facts.append({
                    "claim": sentence,
                    "source": page["url"],
                    "confidence": 0.85,
                    "direct_quote": sentence[:60],
                    "sub_question": page.get("title", ""),
                    "search_type": page.get("search_type", "general"),
                })
        if not facts:
            # Deterministic fallback: the pack for the routed topic, reusing the
            # shared fixture extractor. Never empty (an empty fact list would
            # starve the whole downstream evaluation).
            facts = fixtures_v1.facts_for_pack(fixtures_v1.pages_for_topic(self.topic))
        return {"facts": facts}

    # -- synthesizer ---------------------------------------------------------
    def _golden_answer(self, user_prompt: str) -> Dict[str, Any]:
        numbered = self._parse_evidence(user_prompt)
        answer = self._compose_report(numbered)
        return {"answer": answer}

    @staticmethod
    def _parse_evidence(user_prompt: str) -> List[Dict[str, str]]:
        """Parse `[n] claim ...` evidence lines from the synthesizer prompt.

        The production evidence block is `[n] <claim> (primary, verified, N
        sources agree, direct quote) [angle: ...] [sense: ...]`. The trailing
        machine metadata is stripped so the re-emitted sentence is the bare
        claim (kept verbatim otherwise), which is what the answer-support
        similarity check must score against the verified source claims."""
        out: List[Dict[str, str]] = []
        seen: set = set()
        for line in (user_prompt or "").splitlines():
            match = re.match(r"^\s*\[(\d+)\]\s+(.+?)\s*$", line)
            if not match:
                continue
            number, claim = match.group(1), match.group(2)
            if number in seen:
                continue
            claim = re.sub(r"\s*\[(?:angle|sense):[^\]]*\]\s*$", "", claim)
            claim = re.sub(r"\s*\[(?:angle|sense):[^\]]*\]", " ", claim)
            claim = re.sub(r"\s*\([^()]*(?:primary|verified|sources agree|direct quote)[^()]*\)\s*$", "", claim)
            claim = re.sub(r"\s+", " ", claim).strip()
            if not claim:
                continue
            # Keep the terminal period: verify_answer_support splits on
            # [.!?], so a period-less sentence merges with the following
            # machine section and its numbers fail numeric grounding.
            if not claim.endswith((".", "!", "?")):
                claim += "."
            seen.add(number)
            out.append({"n": number, "claim": claim})
        return out

    @staticmethod
    def _compose_report(numbered: List[Dict[str, str]]) -> str:
        """Deterministic sectioned report citing every emitted claim once.

        Executive Summary and Key Findings carry the prose; the remaining
        mandatory sections are added by the production synthesizer's
        `ensure_required_sections` from measured state if absent. Every
        marker here resolves to a real legend entry, so the citation
        invariants the golden set checks hold by construction."""
        findings = numbered
        if findings:
            opening = f"{findings[0]['claim']} [{findings[0]['n']}]"
        else:
            opening = "The available evidence was insufficient to synthesise a confident answer."
        bullets = [f"- {f['claim']} [{f['n']}]" for f in findings]
        lines = [
            "## Executive Summary",
            "",
            opening,
            "",
            "## Key Findings",
            "",
        ]
        lines.extend(bullets if bullets else ["- No verified findings were extracted."])
        return "\n".join(lines)
