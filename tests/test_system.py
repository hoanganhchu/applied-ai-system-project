"""
Unit tests for the Applied AI Music Recommender System.

Run with: pytest tests/
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retriever import load_catalog, retrieve_candidates
from src.scorer import UserProfile, score_song, rank_songs
from src.agent import run_agent


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


@pytest.fixture
def pop_profile():
    return UserProfile(genre="pop", mood="happy", target_energy=0.85, target_valence=0.85)


@pytest.fixture
def lofi_profile():
    return UserProfile(genre="lofi", mood="chill", target_energy=0.28, target_valence=0.40)


# ── Retriever Tests ───────────────────────────────────────────────────────────

def test_catalog_loads(catalog):
    """Catalog should load with at least 20 songs."""
    assert len(catalog) >= 20


def test_catalog_has_required_fields(catalog):
    """Every song must have required fields."""
    required = {"id", "title", "artist", "genre", "mood", "energy", "valence"}
    for song in catalog:
        assert required.issubset(song.keys()), f"Song missing fields: {song}"


def test_retrieve_energy_filter(catalog):
    """Energy filter should return only songs within range."""
    candidates = retrieve_candidates(catalog, min_energy=0.8, max_energy=1.0)
    for s in candidates:
        assert 0.8 <= s["energy"] <= 1.0, f"Song outside energy range: {s['title']}"


def test_retrieve_fallback_on_narrow_genre(catalog):
    """Unknown genre should trigger fallback and still return candidates."""
    candidates = retrieve_candidates(catalog, genre="jazz", top_k=5)
    assert len(candidates) >= 5


def test_retrieve_returns_minimum_pool(catalog):
    """Retriever should always return at least top_k songs."""
    candidates = retrieve_candidates(catalog, genre="pop", mood="happy", top_k=8)
    assert len(candidates) >= 8


# ── Scorer Tests ──────────────────────────────────────────────────────────────

def test_score_genre_match_adds_points(catalog, pop_profile):
    """A genre-matching song should score higher than a non-matching one."""
    pop_song = next(s for s in catalog if s["genre"] == "pop")
    lofi_song = next(s for s in catalog if s["genre"] == "lofi")
    score_pop = score_song(pop_song, pop_profile).score
    score_lofi = score_song(lofi_song, pop_profile).score
    assert score_pop > score_lofi


def test_score_confidence_in_range(catalog, pop_profile):
    """Confidence should always be between 0 and 1."""
    for song in catalog[:10]:
        result = score_song(song, pop_profile)
        assert 0.0 <= result.confidence <= 1.0, (
            f"Confidence out of range for {song['title']}: {result.confidence}"
        )


def test_score_reasons_non_empty(catalog, pop_profile):
    """Every scored song should have at least one reason."""
    for song in catalog[:5]:
        result = score_song(song, pop_profile)
        assert len(result.reasons) >= 1


def test_rank_returns_top_k(catalog, pop_profile):
    """rank_songs should return exactly top_k results."""
    ranked = rank_songs(catalog, pop_profile, top_k=3)
    assert len(ranked) == 3


def test_rank_is_descending(catalog, lofi_profile):
    """Rankings should be in descending order."""
    ranked = rank_songs(catalog, lofi_profile, top_k=5)
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_invalid_energy_raises():
    """UserProfile should reject energy values outside [0,1]."""
    with pytest.raises(ValueError):
        UserProfile(target_energy=1.5)


# ── Agent Tests ───────────────────────────────────────────────────────────────

def test_agent_returns_recommendations(catalog, pop_profile):
    """Agent should return at least 1 recommendation."""
    result = run_agent(pop_profile, top_k=5, catalog=catalog)
    assert len(result.recommendations) >= 1


def test_agent_steps_are_logged(catalog, pop_profile):
    """Agent result should contain PLAN, ACT, and REFLECT steps."""
    result = run_agent(pop_profile, top_k=5, catalog=catalog)
    step_names = {s.step_name for s in result.steps}
    assert "PLAN" in step_names
    assert "ACT" in step_names
    assert "REFLECT" in step_names


def test_agent_confidence_in_range(catalog, lofi_profile):
    """Agent avg_confidence should be in [0, 1]."""
    result = run_agent(lofi_profile, top_k=5, catalog=catalog)
    assert 0.0 <= result.avg_confidence <= 1.0


def test_agent_unknown_genre_does_not_crash(catalog):
    """Agent should not crash on a genre not in catalog."""
    profile = UserProfile(genre="jazz", mood="chill", target_energy=0.4, target_valence=0.5)
    result = run_agent(profile, top_k=5, catalog=catalog)
    assert result.recommendations is not None


def test_agent_top1_genre_matches_for_clear_profile(catalog):
    """For an unambiguous genre profile, top recommendation should match genre."""
    # Classical is very distinct (low energy, high acousticness)
    profile = UserProfile(
        genre="classical", mood="peaceful",
        target_energy=0.18, target_valence=0.65
    )
    result = run_agent(profile, top_k=5, catalog=catalog)
    top1_genre = result.recommendations[0].song["genre"]
    assert top1_genre == "classical", f"Expected classical, got {top1_genre}"
