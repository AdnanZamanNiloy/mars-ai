"""Golden benchmark v1 offline fixtures: deterministic pages + scripted answers.

This module is the DATA HALF of the golden evaluator. It is imported by
`bench.eval_offline` and consumed by the golden mocks in
`bench/mock_pipeline.py` (GoldenFakeLLM / GoldenFakeSearch), which extend the
existing end-to-end mocks rather than replacing them.

Design constraints
------------------
* Fully deterministic and network-free: no LLM, no HTTP, no clock reads.
* Domain-appropriate pages per topic so the pipeline exercises REAL evidence
  handling (verification, dedupe/corroboration, grading, contradictions,
  citation legends) instead of one canned RAG paragraph for every query.
* Every claim a page yields is a verbatim sentence from that page, so the
  production verifier can check it against its cited source. Corroboration is
  modelled by the SAME sentence appearing (near-verbatim) on pages from two
  distinct registrable domains — dedupe then records it as corroboration.
* Off-topic senses are represented by a `forbidden` keyword list per query
  (e.g. an ML-transformer query must not cite electrical-transformer or
  power-grid sources). The evaluator checks cited source URL/text against it.

A "page" is a search-result-shaped dict. A "topic pack" is a list of pages.
`facts_for_pack` turns a pack into the deterministic fact list the scripted
summarizer emits (one fact per claim sentence, cited to its own page).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

# ---------------------------------------------------------------------------
# Topic packs: domain-appropriate pages, several publishers per topic.
# ---------------------------------------------------------------------------
# Each page's `content` holds 1-2 declarative sentences; the scripted
# summarizer emits sentences of >= MIN_CLAIM_CHARS as claims. Two pages on
# different registrable domains restating the same sentence create the
# corroboration signal the evaluator measures.

def _page(url: str, title: str, content: str, search_type: str = "general",
          published_at: str = "2024-06-15") -> Dict[str, Any]:
    return {
        "url": url,
        "title": title,
        "snippet": content[:180],
        "content": content,
        "published_at": published_at,
        "search_type": search_type,
    }


TOPIC_PACKS: Dict[str, List[Dict[str, Any]]] = {
    # -- retrieval augmented generation --------------------------------------
    "rag": [
        _page(
            "https://arxiv.org/abs/rag-survey",
            "Retrieval Augmented Generation: A Survey",
            "Retrieval augmented generation is a technique that combines external "
            "search with language models. RAG systems retrieve documents at query "
            "time and condition generation on the retrieved passages. Retrieval "
            "augmented generation grounds model outputs in retrieved evidence and "
            "reduces fabrication relative to parametric baselines.",
            "academic", "2024-03-01",
        ),
        _page(
            "https://aclanthology.org/rag-eval",
            "Evaluating Retrieval Augmented Generation",
            "Benchmarks show retrieval augmented generation reduces hallucination "
            "rates by 30 percent relative to parametric baselines. Grounding "
            "accuracy improved from 61 percent to 74 percent on the evaluation "
            "suite. Retrieval quality dominates end-to-end accuracy in RAG systems.",
            "academic", "2024-02-10",
        ),
        _page(
            "https://iea.org/rag-adoption",
            "Enterprise Adoption of AI Retrieval Systems",
            "Retrieval augmented generation grounds model outputs in retrieved "
            "evidence and reduces fabrication relative to parametric baselines. "
            "Enterprise adoption of retrieval systems reached 62 percent of "
            "surveyed companies in 2024.",
            "statistical", "2024-05-20",
        ),
        _page(
            "https://dl.acm.org/rag-critique",
            "Critical Perspectives on RAG Evaluation",
            "Retrieval augmented generation does not eliminate hallucination "
            "entirely. Critics note evaluation setups often overstate RAG gains "
            "because baselines are untuned.",
            "academic", "2024-04-11",
        ),
    ],
    # -- mRNA vaccines -------------------------------------------------------
    "mrna": [
        _page(
            "https://www.who.int/mrna-vaccines",
            "mRNA Vaccines: How They Work",
            "An mRNA vaccine is a vaccine that delivers genetic instructions for "
            "a viral protein into host cells. Cells translate the mRNA into the "
            "antigen, which the immune system recognises and remembers.",
            "general", "2024-01-12",
        ),
        _page(
            "https://pubmed.ncbi.nlm.nih.gov/mrna-review",
            "Mechanism of mRNA Vaccine Immunity",
            "The immune system recognises the translated antigen and mounts a "
            "protective response. Cells translate the mRNA into the antigen, "
            "which the immune system recognises and remembers.",
            "academic", "2023-11-05",
        ),
        _page(
            "https://nejm.org/mrna-trial",
            "Randomized Trial of mRNA Vaccination",
            "In the randomized trial, mRNA vaccination led to a 23 percent "
            "relative reduction in major adverse events compared with placebo. "
            "The platform can be updated rapidly when a pathogen mutates.",
            "academic", "2023-09-18",
        ),
    ],
    # -- transformer neural architecture (off-topic senses matter) -----------
    "transformer_ml": [
        _page(
            "https://arxiv.org/abs/1706.03762",
            "Attention Is All You Need",
            "The Transformer architecture processes all tokens in parallel using "
            "self-attention, unlike recurrent models which operate sequentially. "
            "Self-attention lets every token attend to every other token in the "
            "sequence, enabling substantially faster training.",
            "academic", "2017-06-12",
        ),
        _page(
            "https://aclanthology.org/transformer-study",
            "Understanding Self-Attention",
            "Self-attention lets every token attend to every other token in the "
            "sequence, enabling substantially faster training. The Transformer "
            "architecture is a neural network built entirely on attention.",
            "academic", "2021-08-01",
        ),
        _page(
            "https://dl.acm.org/transformer-survey",
            "A Survey of Transformer Models",
            "The Transformer architecture is a neural network built entirely on "
            "attention. Transformer models replaced recurrence in sequence "
            "modeling and became the basis of modern language models.",
            "academic", "2022-05-30",
        ),
    ],
    # -- solid-state batteries ----------------------------------------------
    "solid_state": [
        _page(
            "https://www.energy.gov/solid-state",
            "Solid-State Batteries Explained",
            "A solid-state battery is a battery that uses a solid electrolyte "
            "instead of the liquid electrolyte found in lithium-ion cells. The "
            "solid electrolyte reduces flammability and can enable higher energy "
            "density.",
            "general", "2024-03-22",
        ),
        _page(
            "https://nature.com/solid-state-review",
            "Solid Electrolytes for Batteries",
            "The solid electrolyte reduces flammability and can enable higher "
            "energy density. A solid-state battery replaces the liquid "
            "electrolyte of conventional cells with a solid conductor.",
            "academic", "2023-12-14",
        ),
    ],
    # -- renewable capacity trend -------------------------------------------
    "renewables": [
        _page(
            "https://iea.org/renewables-2023",
            "Renewable Capacity Additions 2023",
            "Renewable capacity additions worldwide reached 510 GW in 2023, an "
            "increase of almost 50 percent compared with 2022. Solar PV "
            "accounted for three-quarters of the additions, while wind additions "
            "declined slightly.",
            "statistical", "2024-01-11",
        ),
        _page(
            "https://irena.org/renewable-report",
            "Global Renewable Report",
            "Renewable capacity additions worldwide reached 510 GW in 2023, an "
            "increase of almost 50 percent compared with 2022. Global investment "
            "in clean energy reached 1.7 trillion dollars in 2023.",
            "statistical", "2024-02-01",
        ),
        _page(
            "https://ember.org/review",
            "Global Electricity Review",
            "Solar PV accounted for three-quarters of the additions, while wind "
            "additions declined slightly. Wind capacity additions declined in "
            "2023 compared with 2022, affected by supply chain constraints.",
            "statistical", "2024-01-30",
        ),
    ],
    # -- AI enterprise adoption ---------------------------------------------
    "ai_adoption": [
        _page(
            "https://mckinsey.com/ai-state",
            "The State of AI",
            "Enterprise adoption of artificial intelligence reached 62 percent "
            "of surveyed companies in 2024. Adoption nearly doubled year over "
            "year as tooling matured.",
            "statistical", "2024-06-01",
        ),
        _page(
            "https://oecd.org/ai-adoption",
            "AI Adoption Across Firms",
            "Enterprise adoption of artificial intelligence reached 62 percent "
            "of surveyed companies in 2024. Larger firms adopt AI faster than "
            "smaller firms across the sampled economies.",
            "statistical", "2024-05-15",
        ),
        _page(
            "https://arxiv.org/ai-diffusion",
            "Diffusion of AI in Industry",
            "Adoption nearly doubled year over year as tooling matured. Larger "
            "firms adopt AI faster than smaller firms across the sampled "
            "economies.",
            "academic", "2024-04-20",
        ),
    ],
    # -- grid-scale storage --------------------------------------------------
    "storage": [
        _page(
            "https://iea.org/storage-report",
            "Grid-Scale Storage Deployment",
            "Grid-scale battery storage deployment expanded at a record pace in "
            "2023, nearly doubling year over year. Lithium-ion battery pack "
            "prices fell to 139 dollars per kilowatt-hour in 2023.",
            "statistical", "2024-02-20",
        ),
        _page(
            "https://nrel.gov/storage",
            "Utility-Scale Storage Trends",
            "Grid-scale battery storage deployment expanded at a record pace in "
            "2023, nearly doubling year over year. Storage costs continued to "
            "fall as manufacturing scaled.",
            "statistical", "2024-03-05",
        ),
    ],
    # -- nuclear vs solar comparison ----------------------------------------
    "nuclear": [
        _page(
            "https://iea.org/nuclear-cost",
            "Nuclear Power Cost Drivers",
            "Nuclear plants provide firm baseload power but have the highest "
            "levelised cost of new generation among mature technologies. Solar "
            "photovoltaic provides variable output and requires storage for firm "
            "capacity.",
            "statistical", "2024-04-02",
        ),
        _page(
            "https://worldbank.org/nuclear-review",
            "The Economics of Nuclear Energy",
            "Nuclear plants provide firm baseload power but have the highest "
            "levelised cost of new generation among mature technologies. Solar "
            "photovoltaic provides variable output and requires storage for firm "
            "capacity.",
            "statistical", "2023-10-19",
        ),
        _page(
            "https://lazard.com/lcoe",
            "Levelised Cost of Energy Analysis",
            "Utility-scale solar has a lower levelised cost than new nuclear in "
            "most markets, but its capacity factor is around a quarter of a "
            "nuclear plant's.",
            "industry", "2024-06-10",
        ),
    ],
    # -- lithium-ion battery costs ------------------------------------------
    "lithium": [
        _page(
            "https://iea.org/battery-outlook",
            "Battery Price Outlook",
            "Lithium-ion battery pack prices fell to 139 dollars per "
            "kilowatt-hour in 2023, a decline of 14 percent from the previous "
            "year. Manufacturing scale and falling material costs drove the "
            "decline.",
            "statistical", "2024-01-15",
        ),
        _page(
            "https://nature.com/battery-costs",
            "Learning Curves in Battery Costs",
            "Lithium-ion battery pack prices fell to 139 dollars per "
            "kilowatt-hour in 2023, a decline of 14 percent from the previous "
            "year. Learning-by-doing and scale drove the cost reduction.",
            "academic", "2023-11-22",
        ),
    ],
    # -- fine-tuning vs RAG --------------------------------------------------
    "finetune": [
        _page(
            "https://arxiv.org/finetune-vs-rag",
            "Fine-Tuning versus Retrieval",
            "Fine-tuning adapts model weights to a task but cannot update facts "
            "without retraining. Retrieval augmented generation grounds model "
            "outputs in retrieved evidence and reduces fabrication relative to "
            "parametric baselines.",
            "academic", "2023-08-14",
        ),
        _page(
            "https://aclanthology.org/rag-comparison",
            "Comparing Adaptation Strategies",
            "Retrieval augmented generation grounds model outputs in retrieved "
            "evidence and reduces fabrication relative to parametric baselines. "
            "Fine-tuning adapts model weights to a task but cannot update facts "
            "without retraining.",
            "academic", "2023-07-30",
        ),
    ],
    # -- carbon pricing vs subsidies ----------------------------------------
    "carbon_policy": [
        _page(
            "https://oecd.org/carbon-pricing",
            "Carbon Pricing Instruments",
            "Carbon pricing raises the cost of emissions and is a cost-effective "
            "way to cut emissions where coverage is broad. Direct subsidies "
            "accelerate deployment of clean technologies but are fiscally "
            "costly.",
            "policy", "2024-03-12",
        ),
        _page(
            "https://worldbank.org/carbon-subsidies",
            "Subsidies and Carbon Pricing Compared",
            "Carbon pricing raises the cost of emissions and is a cost-effective "
            "way to cut emissions where coverage is broad. Direct subsidies "
            "accelerate deployment but tend to be less efficient per dollar.",
            "policy", "2023-12-01",
        ),
    ],
    # -- heat pumps ----------------------------------------------------------
    "heat_pumps": [
        _page(
            "https://www.energy.gov/heat-pumps",
            "Heat Pump Basics",
            "A heat pump is a device that moves heat from a colder space to a "
            "warmer one rather than generating heat by combustion. Because it "
            "moves rather than creates heat, a heat pump can deliver more energy "
            "as heat than it consumes as electricity.",
            "general", "2024-02-14",
        ),
        _page(
            "https://nrel.gov/heat-pump-efficiency",
            "Heat Pump Efficiency",
            "Because it moves rather than creates heat, a heat pump can deliver "
            "more energy as heat than it consumes as electricity. Modern "
            "cold-climate heat pumps maintain capacity at low outdoor "
            "temperatures.",
            "statistical", "2023-11-30",
        ),
    ],
    # -- electric vehicles ---------------------------------------------------
    "ev": [
        _page(
            "https://iea.org/ev-outlook",
            "Global EV Outlook",
            "Global electric vehicle sales reached 14 million units in 2023, a "
            "35 percent increase over 2022. Electric vehicles accounted for "
            "roughly 18 percent of total car sales in 2023.",
            "statistical", "2024-04-23",
        ),
        _page(
            "https://bnef.com/ev-report",
            "EV Market Trends",
            "Global electric vehicle sales reached 14 million units in 2023, a "
            "35 percent increase over 2022. Growth slowed in some markets as "
            "subsidies were phased out.",
            "industry", "2024-05-02",
        ),
    ],
    # -- wind vs solar -------------------------------------------------------
    "wind_solar": [
        _page(
            "https://lazard.com/wind-solar-lcoe",
            "Wind and Solar Cost Comparison",
            "Utility-scale solar has a lower levelised cost than onshore wind in "
            "most markets. Wind has a higher capacity factor than solar, "
            "producing power for more hours per year.",
            "industry", "2024-06-11",
        ),
        _page(
            "https://nrel.gov/wind-solar",
            "Comparing Wind and Solar Resources",
            "Wind has a higher capacity factor than solar, producing power for "
            "more hours per year. Utility-scale solar has a lower levelised cost "
            "than onshore wind in most markets.",
            "statistical", "2024-01-25",
        ),
        _page(
            "https://irena.org/wind-solar-costs",
            "Renewable Power Generation Costs",
            "Utility-scale solar has a lower levelised cost than onshore wind in "
            "most markets. Combined wind and solar deployment continues to grow "
            "worldwide.",
            "statistical", "2023-08-08",
        ),
    ],
    # -- AI regulation -------------------------------------------------------
    "ai_regulation": [
        _page(
            "https://oecd.org/ai-principles",
            "Principles for AI Governance",
            "Binding regulation can set minimum safety standards for frontier "
            "artificial intelligence models. Industry self-regulation is faster "
            "to update but lacks enforcement mechanisms.",
            "policy", "2024-05-09",
        ),
        _page(
            "https://europa.eu/ai-act",
            "The AI Act Explained",
            "Binding regulation can set minimum safety standards for frontier "
            "artificial intelligence models. Enforcement requires technical "
            "capacity that many regulators currently lack.",
            "policy", "2024-04-15",
        ),
    ],
    # -- ambiguous terms -----------------------------------------------------
    "transformer_ambiguous": [
        _page(
            "https://en.wikipedia.org/transformer",
            "Transformer",
            "The transformer is a neural network architecture built on "
            "self-attention. An electrical transformer is a device that changes "
            "the voltage of alternating current in a power grid.",
            "encyclopedia", "2024-01-01",
        ),
        _page(
            "https://britannica.com/transformer-architecture",
            "Transformer (Machine Learning)",
            "The transformer is a neural network architecture built on "
            "self-attention. Transformer models process all tokens in parallel "
            "using attention.",
            "encyclopedia", "2023-10-01",
        ),
    ],
    "apple": [
        _page(
            "https://en.wikipedia.org/apple",
            "Apple",
            "Apple Inc. is an American technology company that designs consumer "
            "electronics and software. The apple is also the edible fruit "
            "produced by an apple tree.",
            "encyclopedia", "2024-01-01",
        ),
        _page(
            "https://britannica.com/apple-company",
            "Apple Inc.",
            "Apple Inc. is an American technology company that designs consumer "
            "electronics and software. The company was founded in 1976.",
            "encyclopedia", "2023-09-15",
        ),
    ],
    "bank": [
        _page(
            "https://en.wikipedia.org/bank",
            "Bank",
            "A bank is a financial institution that accepts deposits and makes "
            "loans. A river bank is the land alongside a river.",
            "encyclopedia", "2024-01-01",
        ),
        _page(
            "https://britannica.com/bank-finance",
            "Bank (Finance)",
            "A bank is a financial institution that accepts deposits and makes "
            "loans. Banks are regulated by central authorities.",
            "encyclopedia", "2023-08-20",
        ),
    ],
    # -- generic fallback (used only when no topic matches) ------------------
    "generic": [
        _page(
            "https://en.wikipedia.org/general",
            "General Reference",
            "This topic concerns a general research question that has no "
            "dedicated fixture. General reference material describes the subject "
            "at a high level.",
            "encyclopedia", "2024-01-01",
        ),
    ],
}


# ---------------------------------------------------------------------------
# Query id -> topic pack. Explicit so a fixture rename is a data edit.
# ---------------------------------------------------------------------------

TOPIC_BY_QUERY_ID: Dict[str, str] = {
    "factual-rag": "rag",
    "factual-mrna": "mrna",
    "factual-transformer-ml": "transformer_ml",
    "factual-solid-state": "solid_state",
    "trend-renewables": "renewables",
    "trend-ai-adoption": "ai_adoption",
    "trend-storage": "storage",
    "comparison-nuclear-solar": "nuclear",
    "comparison-lithium-solid": "lithium",
    "comparison-rag-finetune": "finetune",
    "decision-nuclear-investment": "nuclear",
    "decision-carbon-policy": "carbon_policy",
    "ambiguous-transformer": "transformer_ambiguous",
    "ambiguous-apple": "apple",
    "ambiguous-bank": "bank",
    "causal-rag-hallucination": "rag",
    "causal-battery-costs": "lithium",
    "causal-solar-growth": "renewables",
    "quant-renewable": "renewables",
    "quant-battery-price": "lithium",
    "quant-ai-investment": "ai_adoption",
    "factual-heat-pumps": "heat_pumps",
    "trend-ev-adoption": "ev",
    "comparison-wind-solar": "wind_solar",
    "causal-nuclear-cost": "nuclear",
    "decision-ai-regulation": "ai_regulation",
}


# Keyword router for the intent grounding search (which runs on the raw query
# before the planner). Longest/most-specific keywords first.
_TOPIC_KEYWORDS: List[tuple] = [
    ("retrieval augmented generation", "rag"),
    ("rag", "rag"),
    ("fine-tuning", "finetune"),
    ("mrna", "mrna"),
    ("transformer neural", "transformer_ml"),
    ("transformer architecture", "transformer_ml"),
    ("solid-state", "solid_state"),
    ("renewable", "renewables"),
    ("solar photovoltaic", "renewables"),
    ("artificial intelligence", "ai_adoption"),
    ("battery storage", "storage"),
    ("grid-scale", "storage"),
    ("nuclear", "nuclear"),
    ("lithium-ion", "lithium"),
    ("carbon pricing", "carbon_policy"),
    ("heat pump", "heat_pumps"),
    ("electric vehicle", "ev"),
    ("wind", "wind_solar"),
    ("solar", "wind_solar"),
    ("regulation", "ai_regulation"),
    ("bank", "bank"),
    ("apple", "apple"),
    ("transformer", "transformer_ambiguous"),
]


def route_topic(text: str) -> str:
    """Deterministic topic router for a raw query / sub-question text."""
    low = (text or "").lower()
    for keyword, topic in _TOPIC_KEYWORDS:
        if keyword in low:
            return topic
    return "generic"


def pages_for_topic(topic: str) -> List[Dict[str, Any]]:
    return [dict(p) for p in TOPIC_PACKS.get(topic, TOPIC_PACKS["generic"])]


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
MIN_CLAIM_CHARS = 50


def claim_sentences(content: str, limit: int = 3) -> List[str]:
    """Deterministic claim extraction: sentence split, length and hygiene
    guard, capped. Mirrors the shape the real summarizer accepts."""
    out: List[str] = []
    for raw in _SENTENCE_SPLIT.split(content or ""):
        sentence = " ".join(raw.split()).strip()
        if len(sentence) < MIN_CLAIM_CHARS:
            continue
        if sentence.lower().startswith(("accept all cookies", "read more")):
            continue
        out.append(sentence)
        if len(out) >= limit:
            break
    return out


def facts_for_pack(pages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic fact list for a page pack: one fact per claim sentence,
    cited to its own page. Duplicated sentences across pages become duplicate
    facts, which the production dedupe merges into a corroborated claim."""
    facts: List[Dict[str, Any]] = []
    for page in pages:
        for sentence in claim_sentences(page.get("content", "")):
            facts.append({
                "claim": sentence,
                "source": page["url"],
                "confidence": 0.85,
                "direct_quote": sentence[:60],
                "sub_question": page.get("title", ""),
                "search_type": page.get("search_type", "general"),
                "published_at": page.get("published_at", "2024-06-15"),
            })
    return facts


def expected_by_id() -> Dict[str, Dict[str, Any]]:
    """The per-id expectation dict, keyed for the evaluator."""
    from bench.golden.loader import load_queries

    return {q["id"]: q for q in load_queries()}
