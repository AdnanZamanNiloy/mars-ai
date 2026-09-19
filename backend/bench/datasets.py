"""Labeled benchmark datasets for the offline MARS evaluation suite.

Every entry is hand-labeled ground truth for one component. Cases are drawn
from realistic research-domain phrasings so scores reflect production
behavior, not toy strings. Domains are real registry entries so the source
authority checks see realistic tiering.

Label conventions:
  expected_verified   — what verify_facts SHOULD conclude (hard checks)
  expected_supported  — what verify_answer_support SHOULD conclude per sentence
  expected_conflict   — the contradiction engine SHOULD flag this pair
  human_similarity    — 0..1 reference affinity for correlation metrics
  is_duplicate        — dedupe_semantic_facts SHOULD merge this pair (0.86)
"""

# ---------------------------------------------------------------------------
# 1. Verification accuracy: (claim, source_url, source_text, expected_verified)
# ---------------------------------------------------------------------------

VERIFICATION_CASES = [
    # --- grounded, should verify -----------------------------------------
    ("Global renewable capacity additions reached 510 GW in 2023.",
     "https://iea.org/renewables-2023",
     "Renewable capacity additions worldwide reached 510 GW in 2023, an increase of almost 50% "
     "compared with 2022. Solar PV accounted for three-quarters of the additions.",
     True),
    ("Solar PV made up roughly three quarters of 2023 capacity additions.",
     "https://iea.org/renewables-2023",
     "Renewable capacity additions worldwide reached 510 GW in 2023. Solar PV accounted for "
     "three-quarters of the additions, with wind and hydro providing most of the remainder.",
     True),
    ("The study reports a 23% relative reduction in major adverse events.",
     "https://nejm.org/trial-results",
     "In the randomized trial, the intervention led to a 23% relative reduction in major "
     "adverse cardiovascular events compared with placebo over the follow-up period.",
     True),
    ("Transformer models process tokens in parallel rather than sequentially.",
     "https://arxiv.org/attention-paper",
     "The Transformer architecture processes all tokens in parallel using self-attention, "
     "unlike recurrent models which operate sequentially, enabling substantially faster training.",
     True),
    ("Wind installations declined in 2023 relative to the prior year.",
     "https://ember.org/review",
     "Wind capacity additions declined in 2023 compared with 2022, affected by supply chain "
     "constraints and slower permitting in several markets.",
     True),
    # --- numeric hallucination, should fail --------------------------------
    ("Global renewable capacity additions reached 620 GW in 2023.",
     "https://iea.org/renewables-2023",
     "Renewable capacity additions worldwide reached 510 GW in 2023, an increase of almost 50% "
     "compared with 2022.",
     False),
    ("The study reports a 41% relative reduction in major adverse events.",
     "https://nejm.org/trial-results",
     "In the randomized trial, the intervention led to a 23% relative reduction in major "
     "adverse cardiovascular events compared with placebo.",
     False),
    # --- polarity inversion, should fail -----------------------------------
    ("Wind installations grew in 2023 relative to the prior year.",
     "https://ember.org/review",
     "Wind capacity additions declined in 2023 compared with 2022, affected by supply chain "
     "constraints and slower permitting.",
     False),
    ("The intervention failed to reduce adverse events compared with placebo.",
     "https://nejm.org/trial-results",
     "In the randomized trial, the intervention led to a 23% relative reduction in major "
     "adverse cardiovascular events compared with placebo over the follow-up period.",
     False),
    # --- topical mismatch, should fail -------------------------------------
    ("Electric vehicle sales in China reached 8 million units.",
     "https://iea.org/renewables-2023",
     "Renewable capacity additions worldwide reached 510 GW in 2023. Solar PV accounted for "
     "three-quarters of the additions.",
     False),
    # --- unit mismatch, should fail ----------------------------------------
    ("Battery costs fell to 95 dollars per barrel.",
     "https://iea.org/battery-outlook",
     "Lithium-ion battery pack prices fell to 139 dollars per kilowatt-hour in 2023, a decline "
     "of 14% from the previous year.",
     False),
    # --- grounded with scale folding, should verify -------------------------
    ("Renewable investment totaled approximately 1.7 trillion dollars.",
     "https://iea.org/world-energy-investment",
     "Global investment in clean energy reached 1.7 trillion USD in 2023, while fossil fuel "
     "investment was slightly above 1 trillion dollars.",
     True),
    # --- grounded percentage, should verify ---------------------------------
    ("Battery pack prices declined by 14% in 2023.",
     "https://iea.org/battery-outlook",
     "Lithium-ion battery pack prices fell to 139 dollars per kilowatt-hour in 2023, a decline "
     "of 14% from the previous year.",
     True),
    # --- close-but-wrong percentage, should fail ----------------------------
    ("Battery pack prices declined by 40% in 2023.",
     "https://iea.org/battery-outlook",
     "Lithium-ion battery pack prices fell to 139 dollars per kilowatt-hour in 2023, a decline "
     "of 14% from the previous year.",
     False),
    # --- paraphrase grounded, should verify ----------------------------------
    ("Grid-scale storage deployment expanded at a record pace last year.",
     "https://iea.org/storage-report",
     "Grid-scale battery storage deployment expanded at a record pace in 2023, nearly doubling "
     "year over year as costs continued to fall.",
     True),
    # --- source missing content, should fail --------------------------------
    ("Any claim at all worth checking.",
     "https://iea.org/empty-page",
     "",
     False),
]

# ---------------------------------------------------------------------------
# 2. Hallucination adversarial set: claims a model might fabricate.
#     Expected: REJECTED (not verified). false_accept = hallucination leak.
# ---------------------------------------------------------------------------

HALLUCINATION_CASES = [
    ("The trial enrolled exactly 12483 participants across 14 countries.",
     "https://nejm.org/trial-results",
     "In the randomized trial, the intervention led to a 23% relative reduction in major "
     "adverse cardiovascular events compared with placebo over the follow-up period."),
    ("Renewable capacity additions reached 510 GW in 2023 and created 2.3 million jobs.",
     "https://iea.org/renewables-2023",
     "Renewable capacity additions worldwide reached 510 GW in 2023, an increase of almost 50% "
     "compared with 2022. Solar PV accounted for three-quarters of the additions."),
    ("The paper was authored by 42 researchers and cited 900 prior works.",
     "https://arxiv.org/attention-paper",
     "The Transformer architecture processes all tokens in parallel using self-attention."),
    ("Grid-scale storage costs are projected to reach 12 dollars per kWh by 2026.",
     "https://iea.org/storage-report",
     "Grid-scale battery storage deployment expanded at a record pace in 2023, nearly doubling "
     "year over year as costs continued to fall."),
    ("The study was later retracted for data irregularities.",
     "https://nejm.org/trial-results",
     "In the randomized trial, the intervention led to a 23% relative reduction in major "
     "adverse cardiovascular events compared with placebo."),
    ("Wind capacity declined because turbines became 15% less efficient.",
     "https://ember.org/review",
     "Wind capacity additions declined in 2023 compared with 2022, affected by supply chain "
     "constraints and slower permitting in several markets."),
]

# ---------------------------------------------------------------------------
# 3. Contradiction detection: (claim_a, claim_b, source_a, source_b, expected_conflict)
# ---------------------------------------------------------------------------

CONTRADICTION_PAIRS = [
    ("The market grew by 25% last year", "The market grew by 11% last year",
     "https://a.com/x", "https://b.com/y", True),     # numeric
    ("The new model outperforms the baseline", "The new model does not outperform the baseline",
     "https://a.com/x", "https://b.com/y", True),     # polarity
    ("Capacity reached 1200 GW in 2023", "Capacity reached 1600 GW in 2024",
     "https://a.com/x", "https://b.com/y", True),     # temporal
    ("Inflation rose to 9 percent in the eurozone", "Eurozone inflation climbed to 9% this year",
     "https://a.com/x", "https://b.com/y", False),    # agreeing restatement
    ("Module efficiency improved by 22%", "New plants added 22 GW",
     "https://a.com/x", "https://b.com/y", False),    # unit mismatch, not a conflict
    ("The company employs 40000 people", "The company employs 40000 people",
     "https://a.com/x", "https://b.com/y", False),    # identical → dedup territory
    ("GDP contracted by 3% in the quarter", "GDP expanded by 2% in the quarter",
     "https://a.com/x", "https://b.com/y", True),     # direction + numeric
    ("The drug is approved for chronic use", "The drug is not approved for chronic use",
     "https://a.com/x", "https://b.com/y", True),     # polarity
    ("Emissions fell 5% in 2022", "Emissions fell 5% in 2022 according to the inventory",
     "https://a.com/x", "https://b.com/y", False),    # agreeing restatement
    ("Sales reached 2.4 billion dollars", "Sales reached 2.4 billion dollars in 2023",
     "https://a.com/x", "https://b.com/y", False),    # agreeing + year context
    ("The reactor produced 12 TWh annually", "The reactor produced 19 TWh annually",
     "https://a.com/x", "https://b.com/y", True),     # numeric same unit
    ("Unemployment fell to 4.1%", "Unemployment fell to 4.2%",
     "https://a.com/x", "https://b.com/y", False),    # within tolerance
]

# ---------------------------------------------------------------------------
# 4. Semantic similarity reference pairs: (a, b, human_label 0..1)
# ---------------------------------------------------------------------------

SIMILARITY_PAIRS = [
    ("Solar capacity grew 40% in 2024", "Solar capacity grew 40% in 2024", 1.0),
    ("Global solar capacity grew by 40% in 2024", "Global solar capacity grew 40% in 2024", 0.95),
    ("Wind and solar capacity roughly doubled over the past decade",
     "Capacity from wind and solar nearly doubled in the last ten years", 0.85),
    ("The central bank raised interest rates by 75 basis points",
     "The monetary authority increased its policy rate by three quarters of a percent", 0.75),
    ("Battery pack prices fell 14% in 2023", "Lithium-ion battery costs declined sharply", 0.6),
    ("Solar capacity grew 40% in 2024", "Wind capacity grew 40% in 2024", 0.75),  # band: same form, different subject
    ("Renewables grew 40% in 2024", "The chef prepared a rustic ragu for dinner", 0.05),
    ("The central bank raised interest rates by 75 basis points",
     "Pigeons are birds commonly found in cities", 0.05),
    ("Transformer models use self-attention", "Recurrent networks process tokens sequentially", 0.35),
    ("The trial showed a 23% reduction in events", "The study found events fell by about a quarter", 0.8),
    ("Emissions trading covers industry", "Cap-and-trade schemes regulate industrial emissions", 0.8),
    ("Grid batteries doubled year over year", "Storage deployment nearly doubled annually", 0.45),  # different subjects, shared trend word
    # Negation pairs: topical affinity yes, but they must sit INSIDE the
    # contradiction band (0.50-0.86), never above it — the engine's operational
    # contract — so the reference label reflects band placement, not raw affinity.
    ("The model outperforms the baseline", "The model does not outperform the baseline", 0.65),
    ("Interest rates rose", "Interest rates fell", 0.6),
    ("Investment reached 1.7 trillion dollars", "Investment totaled 1.7 trillion USD", 0.95),
    ("Coal generation declined in Europe", "European coal power fell", 0.8),
    ("Heat pumps cut heating emissions", "Electrified heating reduces carbon output", 0.7),
    ("The algorithm runs in linear time", "The procedure completes in constant time", 0.6),
    ("Perovskite cells reached 26% efficiency", "Silicon cells reached 26% efficiency", 0.75),  # band: same form, different subject
    ("Demand response shaved peak load", "Demand-side management trimmed consumption peaks", 0.7),
]

# ---------------------------------------------------------------------------
# 5. Dedup pairs: (a, b, is_duplicate at the 0.86 threshold)
# ---------------------------------------------------------------------------

DEDUP_PAIRS = [
    ("Global solar capacity grew by 40% in 2024", "Global solar capacity grew 40% in 2024", True),
    ("Renewable capacity additions reached 510 GW in 2023",
     "Renewable capacity additions worldwide reached 510 GW in 2023", True),
    ("The company employs 40000 people across 12 countries",
     "The company employs 40000 people across 12 nations", True),
    ("Battery pack prices declined 14% in 2023", "Battery pack prices declined by 14% during 2023", True),
    ("Wind and solar capacity roughly doubled over the past decade",
     "Capacity from wind and solar nearly doubled in the last ten years", True),
    ("Solar capacity grew 40% in 2024", "Wind capacity grew 40% in 2024", False),
    ("The central bank raised interest rates", "Pigeons are common city birds", False),
    ("Transformer models use self-attention", "Recurrent networks process sequentially", False),
    ("Battery pack prices fell 14%", "Lithium-ion battery costs declined sharply", False),
    ("The trial showed a 23% reduction", "The study found events fell by a quarter", False),
]

# ---------------------------------------------------------------------------
# 6. Citation support: answer + facts + per-sentence expected verdicts
# ---------------------------------------------------------------------------

CITATION_ANSWER = (
    "Renewable capacity additions reached 510 GW in 2023 with solar accounting for "
    "three quarters of new capacity [1]. "
    "Wind additions declined year over year [2]. "
    "Battery pack prices fell to 139 dollars per kilowatt-hour [3]. "
    "The moon landing was staged in a film studio [1]. "
    "Investment in clean energy reached 1.7 trillion dollars [2]."
    "\n\nSources:\n"
    "[1] iea.org — https://iea.org/renewables-2023\n"
    "[2] ember.org — https://ember.org/review\n"
    "[3] iea.org — https://iea.org/battery-outlook"
)

CITATION_FACTS = [
    {"claim": "Renewable capacity additions worldwide reached 510 GW in 2023",
     "source": "https://iea.org/renewables-2023", "verified": True, "confidence": 0.9},
    {"claim": "Solar PV accounted for three-quarters of the renewable capacity additions",
     "source": "https://iea.org/renewables-2023", "verified": True, "confidence": 0.9},
    {"claim": "Wind capacity additions declined in 2023 compared with 2022",
     "source": "https://ember.org/review", "verified": True, "confidence": 0.85},
    {"claim": "Wind capacity additions declined in 2023 compared with 2022",
     "source": "https://iea.org/renewables-2023", "verified": True, "confidence": 0.85},
    {"claim": "Lithium-ion battery pack prices fell to 139 dollars per kilowatt-hour in 2023",
     "source": "https://iea.org/battery-outlook", "verified": True, "confidence": 0.9},
    {"claim": "Global investment in clean energy reached 1.7 trillion USD in 2023",
     "source": "https://ember.org/review", "verified": True, "confidence": 0.85},
]

# (sentence_index, expected_supported)
CITATION_SENTENCE_LABELS = [
    (0, True),   # 510 GW + solar share — grounded
    (1, True),   # wind decline — grounded
    (2, True),   # battery prices — grounded
    (3, False),  # moon landing — fabricated, cited to wrong source
    (4, True),   # 1.7T investment — grounded via corroborating claim
]

# ---------------------------------------------------------------------------
# 7. Confidence calibration scenarios: designed evidence pools with
#     expected confidence bands.
# ---------------------------------------------------------------------------


def _fact(claim, source, verified=True, score=0.8):
    return {"claim": claim, "source": source, "verified": verified,
            "verification_score": score, "confidence": 0.85}


CALIBRATION_SCENARIOS = [
    {
        "name": "strong",
        "expected_band": (0.70, 1.00),
        "facts": [
            _fact("Renewable additions reached 510 GW in 2023", "https://iea.org/a"),
            _fact("Renewable additions reached 510 GW in 2023", "https://irena.org/b"),
            _fact("Solar accounted for three quarters of additions", "https://iea.org/a"),
            _fact("Wind additions declined slightly", "https://ember.org/c"),
            _fact("Investment reached 1.7 trillion dollars", "https://worldbank.org/d"),
            _fact("Battery prices fell 14 percent", "https://iea.org/battery"),
        ],
        "critique": {"is_sufficient": True},
        "contradictions": [],
    },
    {
        "name": "moderate",
        "expected_band": (0.45, 0.75),
        "facts": [
            _fact("Renewable additions reached 510 GW in 2023", "https://iea.org/a", score=0.55),
            _fact("Solar share rose", "https://irena.org/b", score=0.5),
            _fact("Wind declined", "https://ember.org/c", score=0.5),
        ],
        "critique": {"is_sufficient": False},
        "contradictions": [],
    },
    {
        "name": "weak",
        "expected_band": (0.0, 0.45),
        "facts": [
            _fact("Something unverifiable happened", "https://blogspot.com/x", verified=False, score=0.1),
            _fact("Another unsupported line", "https://weebly.com/y", verified=False, score=0.1),
        ],
        "critique": {"is_sufficient": False},
        "contradictions": [],
    },
    {
        "name": "conflicted",
        "expected_band": (0.0, 0.80),
        "facts": [
            _fact("The market grew by 25% last year", "https://iea.org/a"),
            _fact("The market grew by 5% last year", "https://irena.org/b"),
            _fact("Investment rose overall", "https://worldbank.org/d"),
        ],
        "critique": {"is_sufficient": False},
        "contradictions": [
            {"kind": "numeric", "severity": 0.91, "intra_source": False},
        ],
        "penalty_expected": True,
    },
]

# ---------------------------------------------------------------------------
# 8. Query-router routing: (query, expected_path, reason_tokens)
#     expected_path is what decide_route SHOULD conclude. Stable general
#     knowledge MAY be answered directly; anything depending on time,
#     sourced numbers, decisions, contested topics, or ambiguity MUST be
#     researched. reason_tokens are substrings that must appear in the
#     decision's signals' hard_blockers (or "clear" for an unblocked query).
# ---------------------------------------------------------------------------

ROUTER_ROUTING_CASES = [
    # --- must research: freshness ---
    ("What is the latest price of gold?", "research", ["freshness"]),
    ("How many electric vehicles were sold in 2025?", "research", ["freshness"]),
    ("What is the current population of Bangladesh?", "research", ["freshness"]),
    ("Which company has the highest market cap right now?", "research", ["freshness"]),
    ("What happened in the AI industry this year?", "research", ["freshness"]),
    # --- must research: quantitative ---
    ("What is the market size of the battery industry?", "research", ["quantitative"]),
    ("What percentage of energy comes from solar in Germany?", "research", ["quantitative"]),
    ("How much did global EV sales grow last year?", "research", ["quantitative"]),
    # --- must research: decision framing ---
    ("Should we invest in nuclear energy for our grid?", "research", ["decision"]),
    ("Which database should I choose for this project?", "research", ["decision"]),
    # --- must research: contested / high-stakes ---
    ("Is this supplement safe to take daily?", "research", ["contested"]),
    ("Is the keto diet harmful?", "research", ["contested"]),
    # --- must research: query type (comparative/analytical/exploratory) ---
    ("Compare solar and nuclear energy costs", "research", ["query_type"]),
    ("What is the impact of remote work on productivity?", "research", ["query_type"]),
    ("Give me an overview of the quantum computing landscape", "research", ["query_type"]),
    # --- must research: ambiguity (intent supplied below) ---
    ("What is transformer?", "research", ["ambiguity"], {"ambiguity": True, "query_type": "factual"}),
    ("What is apple?", "research", ["ambiguity"], {"ambiguity": True, "query_type": "factual"}),
    # --- may answer directly: stable general knowledge (no hard blocker) ---
    ("What is a for-loop?", "clear", ["clear"]),
    ("What is a Python list comprehension?", "clear", ["clear"]),
    ("Explain how TCP handles packet loss.", "clear", ["clear"]),
    ("What is the capital of France?", "clear", ["clear"]),
]

