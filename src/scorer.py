"""
scorer.py — Content-based scoring engine for the Applied AI Music Recommender.

Scores each candidate song against a user profile using weighted attribute
matching. Returns a numeric score plus human-readable explanation reasons.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """
    Represents a user's music preferences for scoring.

    Weights control how much each attribute influences the final score.
    All weights default to 1.0 (equal importance).
    """
    genre: str = "pop"
    mood: str = "happy"
    target_energy: float = 0.7
    target_valence: float = 0.7

    genre_weight: float = 2.0
    mood_weight: float = 1.5
    energy_weight: float = 2.0
    valence_weight: float = 1.5

    use_mood: bool = True

    def __post_init__(self):
        # Validate energy and valence are in [0, 1]
        for attr, val in [("target_energy", self.target_energy), ("target_valence", self.target_valence)]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{attr} must be between 0.0 and 1.0, got {val}")


@dataclass
class ScoredSong:
    """A song with its computed score and explanation."""
    song: dict
    score: float
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0


def score_song(song: dict, profile: UserProfile) -> ScoredSong:
    """
    Score a single song against a user profile.

    Scoring components:
    - Genre match: +genre_weight (exact match)
    - Mood match: +mood_weight (exact match, if use_mood=True)
    - Energy proximity: +energy_weight * (1 - |song_energy - target_energy|)
    - Valence proximity: +valence_weight * (1 - |song_valence - target_valence|)

    Args:
        song: Song dict from catalog.
        profile: UserProfile with preferences and weights.

    Returns:
        ScoredSong with numeric score and reasons list.
    """
    total = 0.0
    reasons = []

    # Genre
    if song["genre"].lower() == profile.genre.lower():
        total += profile.genre_weight
        reasons.append(f"genre match ({song['genre']}) +{profile.genre_weight:.1f}")
    else:
        reasons.append(f"genre mismatch ({song['genre']} ≠ {profile.genre})")

    # Mood
    if profile.use_mood:
        if song["mood"].lower() == profile.mood.lower():
            total += profile.mood_weight
            reasons.append(f"mood match ({song['mood']}) +{profile.mood_weight:.1f}")
        else:
            reasons.append(f"mood mismatch ({song['mood']} ≠ {profile.mood})")

    # Energy proximity
    energy_sim = 1.0 - abs(song["energy"] - profile.target_energy)
    energy_contribution = profile.energy_weight * energy_sim
    total += energy_contribution
    reasons.append(
        f"energy sim {energy_sim:.2f} → +{energy_contribution:.2f}"
    )

    # Valence proximity
    valence_sim = 1.0 - abs(song["valence"] - profile.target_valence)
    valence_contribution = profile.valence_weight * valence_sim
    total += valence_contribution
    reasons.append(
        f"valence sim {valence_sim:.2f} → +{valence_contribution:.2f}"
    )

    # Confidence: proportion of max possible score achieved
    max_score = (
        profile.genre_weight
        + (profile.mood_weight if profile.use_mood else 0)
        + profile.energy_weight
        + profile.valence_weight
    )
    confidence = round(total / max_score, 3) if max_score > 0 else 0.0

    logger.debug(f"Scored '{song['title']}': {total:.3f} (confidence={confidence})")

    return ScoredSong(
        song=song,
        score=round(total, 3),
        reasons=reasons,
        confidence=confidence,
    )


def rank_songs(
    candidates: list[dict],
    profile: UserProfile,
    top_k: int = 5,
) -> list[ScoredSong]:
    """
    Score all candidates and return top-K ranked results.

    Args:
        candidates: List of song dicts from retriever.
        profile: UserProfile for scoring.
        top_k: Number of top results to return.

    Returns:
        Sorted list of ScoredSong (highest score first).
    """
    scored = [score_song(s, profile) for s in candidates]
    ranked = sorted(scored, key=lambda x: x.score, reverse=True)
    top = ranked[:top_k]
    logger.info(
        f"Ranked {len(candidates)} songs → top {top_k}: "
        + ", ".join(f"'{s.song['title']}' ({s.score})" for s in top)
    )
    return top
