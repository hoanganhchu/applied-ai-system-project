"""
agent.py — Agentic workflow for the Applied AI Music Recommender.

Implements a 3-step agentic loop:
  1. PLAN   — Decide retrieval strategy from user profile
  2. ACT    — Retrieve candidates + score them
  3. REFLECT — Check output quality; retry or adjust if needed

This adds observable multi-step reasoning with intermediate steps logged at
each stage, satisfying the Agentic Workflow requirement.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .retriever import load_catalog, retrieve_candidates, retrieval_summary
from .scorer import UserProfile, ScoredSong, rank_songs

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """A single observable step in the agent's reasoning chain."""
    step_name: str
    description: str
    data: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    """Full result of the agentic recommendation workflow."""
    profile: UserProfile
    steps: list[AgentStep]
    recommendations: list[ScoredSong]
    avg_confidence: float
    passed_quality_check: bool
    retry_count: int = 0


# Quality threshold: average confidence of top-5 must exceed this
CONFIDENCE_THRESHOLD = 0.45
MAX_RETRIES = 1


def _plan(profile: UserProfile) -> dict:
    """
    PLAN step: decide the retrieval strategy.
    Returns retrieval kwargs based on profile analysis.
    """
    # Determine energy band (±0.25 around target, clamped to [0,1])
    min_e = max(0.0, profile.target_energy - 0.25)
    max_e = min(1.0, profile.target_energy + 0.25)

    plan = {
        "genre": profile.genre,
        "mood": profile.mood if profile.use_mood else None,
        "min_energy": min_e,
        "max_energy": max_e,
    }
    logger.info(f"[PLAN] Retrieval strategy: {plan}")
    return plan


def _act(catalog: list[dict], plan: dict, top_k: int, profile: UserProfile) -> tuple[list, list[ScoredSong]]:
    """
    ACT step: retrieve candidates and score them.
    Returns (candidates, ranked_songs).
    """
    candidates = retrieve_candidates(catalog, **plan, top_k=top_k + 5)
    ranked = rank_songs(candidates, profile, top_k=top_k)
    logger.info(f"[ACT] Retrieved {len(candidates)} candidates, ranked top {top_k}")
    return candidates, ranked


def _reflect(ranked: list[ScoredSong]) -> tuple[bool, float]:
    """
    REFLECT step: evaluate output quality.
    Returns (passed: bool, avg_confidence: float).
    """
    if not ranked:
        return False, 0.0
    avg_conf = sum(s.confidence for s in ranked) / len(ranked)
    passed = avg_conf >= CONFIDENCE_THRESHOLD
    logger.info(
        f"[REFLECT] Avg confidence: {avg_conf:.3f} | "
        f"Threshold: {CONFIDENCE_THRESHOLD} | Passed: {passed}"
    )
    return passed, round(avg_conf, 3)


def run_agent(
    profile: UserProfile,
    top_k: int = 5,
    catalog: Optional[list[dict]] = None,
) -> AgentResult:
    """
    Run the full Plan → Act → Reflect agentic workflow.

    If the Reflect step fails (low confidence), the agent retries with
    relaxed retrieval filters (broader energy band, no mood filter).

    Args:
        profile: UserProfile with user preferences.
        top_k: Number of recommendations to return.
        catalog: Pre-loaded catalog (loads from disk if None).

    Returns:
        AgentResult with all observable steps and final recommendations.
    """
    if catalog is None:
        catalog = load_catalog()

    steps: list[AgentStep] = []
    retry_count = 0

    # ── STEP 1: PLAN ──────────────────────────────────────────────
    plan = _plan(profile)
    steps.append(AgentStep(
        step_name="PLAN",
        description=(
            f"Retrieval strategy decided: genre='{plan['genre']}', "
            f"mood='{plan['mood']}', energy=[{plan['min_energy']:.2f}, {plan['max_energy']:.2f}]"
        ),
        data=plan,
    ))

    # ── STEP 2: ACT ───────────────────────────────────────────────
    candidates, ranked = _act(catalog, plan, top_k, profile)
    steps.append(AgentStep(
        step_name="ACT",
        description=retrieval_summary(candidates),
        data={"candidate_count": len(candidates), "top_k": top_k},
    ))

    # ── STEP 3: REFLECT ───────────────────────────────────────────
    passed, avg_conf = _reflect(ranked)
    steps.append(AgentStep(
        step_name="REFLECT",
        description=(
            f"Quality check {'PASSED' if passed else 'FAILED'}. "
            f"Avg confidence: {avg_conf:.3f} (threshold={CONFIDENCE_THRESHOLD})"
        ),
        data={"avg_confidence": avg_conf, "passed": passed},
    ))

    # ── RETRY if quality check fails ──────────────────────────────
    if not passed and retry_count < MAX_RETRIES:
        retry_count += 1
        relaxed_plan = {
            "genre": profile.genre,
            "mood": None,           # drop mood filter
            "min_energy": 0.0,      # open energy band
            "max_energy": 1.0,
        }
        logger.info(f"[RETRY {retry_count}] Relaxing filters: {relaxed_plan}")
        steps.append(AgentStep(
            step_name=f"RETRY-{retry_count}",
            description=(
                "Confidence below threshold. Relaxing retrieval: "
                "removing mood filter and opening energy band."
            ),
            data=relaxed_plan,
        ))

        candidates, ranked = _act(catalog, relaxed_plan, top_k, profile)
        passed, avg_conf = _reflect(ranked)

        steps.append(AgentStep(
            step_name="REFLECT-2",
            description=(
                f"Post-retry quality check {'PASSED' if passed else 'FAILED'}. "
                f"Avg confidence: {avg_conf:.3f}"
            ),
            data={"avg_confidence": avg_conf, "passed": passed},
        ))

    return AgentResult(
        profile=profile,
        steps=steps,
        recommendations=ranked,
        avg_confidence=avg_conf,
        passed_quality_check=passed,
        retry_count=retry_count,
    )
