"""Self-Diagnosis (4.2): aggregate correctness over fabricated row-dicts.

The script itself is I/O over the real DB and is verified live; pytest
covers the pure analysis in app/core/diagnose.py only.
"""

import asyncio

import pytest

from app.core.diagnose import (
    attention_flags,
    contradiction_watch,
    critic_efficiency,
    run_health,
    verification_by_domain,
)


def _domain(url):
    if "://" in url:
        return url.split("://", 1)[1].split("/", 1)[0].lower()
    return ""


def test_verification_by_domain_groups_and_sorts():
    claims = [
        {"source_url": "https://a.com/1", "confidence": 0.9, "verified": 1},
        {"source_url": "https://a.com/2", "confidence": 0.5, "verified": 0},
        {"source_url": "https://b.org/x", "confidence": 0.8, "verified": 1},
        {"source_url": "", "confidence": 0.1, "verified": 0},
    ]
    rows = verification_by_domain(claims, _domain)
    assert rows[0]["domain"] == "a.com"
    assert rows[0]["claims"] == 2
    assert rows[0]["verified_rate"] == 0.5
    assert rows[0]["avg_confidence"] == 0.7
    assert rows[-1]["domain"] == "(unsourced)"


def test_critic_efficiency_counts_expansions_and_gains():
    reviews = [
        {"run_id": "r1", "iteration": 1, "confidence": 0.5},
        {"run_id": "r1", "iteration": 2, "confidence": 0.8},
        {"run_id": "r2", "iteration": 1, "confidence": 0.9},
    ]
    eff = critic_efficiency(reviews)
    assert eff["runs"] == 2
    assert eff["avg_iterations"] == 1.5
    assert eff["expansion_rate"] == 0.5
    assert eff["avg_confidence_gain"] == pytest.approx(0.3)


def test_critic_efficiency_empty():
    assert critic_efficiency([])["runs"] == 0


def test_run_health_mixes_statuses():
    runs = [
        {"status": "completed", "confidence": 0.8, "estimated_cost": 0.001},
        {"status": "completed", "confidence": 0.6, "estimated_cost": 0.003},
        {"status": "failed", "confidence": 0.0, "estimated_cost": 0.0},
        {"status": "timeout", "confidence": 0.0, "estimated_cost": 0.0},
    ]
    health = run_health(runs)
    assert health["total"] == 4
    assert health["by_status"] == {"completed": 2, "failed": 1, "timeout": 1}
    assert health["fail_rate"] == 0.5
    assert health["avg_confidence"] == 0.7
    assert health["avg_cost"] == 0.002


def test_contradiction_watch_counts_sections():
    watch = contradiction_watch(["# Final Answer\nx\n\n# Contradictions\n- a", "# Final Answer\ny"])
    assert watch == {"reports": 2, "with_contradictions": 1, "rate": 0.5}


def test_attention_flags_conservative():
    health = {"total": 10, "fail_rate": 0.3, "avg_cost": 0.0004, "by_status": {}}
    domains = [{"domain": "spam.io", "claims": 12, "verified": 2, "verified_rate": 2 / 12, "avg_confidence": 0.4}]
    critic = {"runs": 10, "avg_iterations": 2.5, "expansion_rate": 0.9, "avg_confidence_gain": 0.01}
    flags = attention_flags(health, domains, critic)
    assert len(flags) == 3
    assert any("spam.io" in f for f in flags)


def test_attention_flags_quiet_on_healthy():
    health = {"total": 10, "fail_rate": 0.0, "avg_cost": 0.0004, "by_status": {}}
    assert attention_flags(health, [], {"runs": 2, "expansion_rate": 0.0, "avg_confidence_gain": 0.0}) == []


def test_diagnosis_connection_is_read_only(tmp_path):
    """The script's mode=ro handle must refuse writes, not just avoid them."""
    import aiosqlite

    db_path = str(tmp_path / "ro.db")

    async def _check():
        async with aiosqlite.connect(db_path) as db:
            await db.execute("CREATE TABLE t (id INTEGER)")
            await db.commit()
        async with aiosqlite.connect(f"file:{db_path}?mode=ro", uri=True) as ro:
            await ro.execute("SELECT * FROM t")
            try:
                await ro.execute("INSERT INTO t (id) VALUES (1)")
                await ro.commit()
            except Exception:
                return True
            return False

    assert asyncio.run(_check()) is True


def test_recommendations_mirror_flags():
    from app.core.diagnose import recommend

    health = {"total": 10, "fail_rate": 0.3, "avg_cost": 0.02, "by_status": {}}
    domains = [{"domain": "spam.io", "claims": 12, "verified": 2, "verified_rate": 2 / 12, "avg_confidence": 0.4}]
    critic = {"runs": 10, "avg_iterations": 2.5, "expansion_rate": 0.9, "avg_confidence_gain": 0.01}
    recs = recommend(health, domains, critic)
    assert len(recs) == 4
    assert all(set(r) == {"problem", "suggestion"} for r in recs)
    assert any("LOW_QUALITY_DOMAINS" in r["suggestion"] for r in recs)
    assert any("sufficiency_threshold" in r["suggestion"] for r in recs)


def test_recommendations_empty_when_healthy():
    from app.core.diagnose import recommend

    health = {"total": 10, "fail_rate": 0.0, "avg_cost": 0.0004, "by_status": {}}
    assert recommend(health, [], {"runs": 2, "expansion_rate": 0.0, "avg_confidence_gain": 0.0}) == []
