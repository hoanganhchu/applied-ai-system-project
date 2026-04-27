"""
retriever.py — RAG component for the Applied AI Music Recommender System.

Retrieves a relevant candidate subset from the song catalog BEFORE the AI
scoring agent runs, reducing the search space and grounding the agent's
recommendations in real catalog data.

This is the "Retrieval" step in Retrieval-Augmented Generation (RAG).
"""

import csv
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")


def load_catalog(path: str = DATA_PATH) -> list[dict]:
    """Load all songs from CSV into a list of dicts."""
    songs = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Cast numeric fields
                for field in ("energy", "valence", "danceability", "acousticness"):
                    row[field] = float(row[field])
                row["tempo_bpm"] = int(row["tempo_bpm"])
                row["id"] = int(row["id"])
                songs.append(row)
        logger.info(f"Catalog loaded: {len(songs)} songs from {path}")
    except FileNotFoundError:
        logger.error(f"Catalog file not found: {path}")
        raise
    return songs


def retrieve_candidates(
    catalog: list[dict],
    genre: Optional[str] = None,
    mood: Optional[str] = None,
    min_energy: float = 0.0,
    max_energy: float = 1.0,
    top_k: int = 12,
) -> list[dict]:
    """
    RAG retrieval step: filter catalog to a relevant candidate subset.

    Applies hard filters first (genre, mood, energy range), then falls back
    to relaxed matching if too few candidates are found.

    Fallback only relaxes soft constraints (genre/mood). Hard constraints such
    as the requested energy range are always preserved.

    Args:
        catalog: Full song list from load_catalog()
        genre: Preferred genre string (e.g. "pop"). None = no filter.
        mood: Preferred mood string (e.g. "happy"). None = no filter.
        min_energy: Minimum energy value (0.0-1.0).
        max_energy: Maximum energy value (0.0-1.0).
        top_k: Minimum number of candidates to return.

    Returns:
        Filtered list of song dicts.
    """
    candidates = catalog[:]

    # --- Hard filter: energy range ---
    candidates = [
        s for s in candidates if min_energy <= s["energy"] <= max_energy
    ]
    logger.debug(f"After energy filter [{min_energy},{max_energy}]: {len(candidates)} songs")

    # --- Soft filter: genre match (with fallback) ---
    if genre:
        genre_matches = [s for s in candidates if s["genre"].lower() == genre.lower()]
        if len(genre_matches) >= 3:
            candidates = genre_matches
            logger.debug(f"Genre filter '{genre}': {len(candidates)} songs")
        else:
            logger.debug(f"Genre filter '{genre}' too narrow ({len(genre_matches)}), keeping all")

    # --- Soft filter: mood match (with fallback) ---
    if mood:
        mood_matches = [s for s in candidates if s["mood"].lower() == mood.lower()]
        if len(mood_matches) >= 2:
            candidates = mood_matches
            logger.debug(f"Mood filter '{mood}': {len(candidates)} songs")
        else:
            logger.debug(f"Mood filter '{mood}' too narrow ({len(mood_matches)}), keeping all")

    # --- Fallback: ensure minimum pool size when soft filters are too narrow ---
    # Keep hard constraints (like energy-only queries) intact. We only relax when
    # genre and/or mood filters were requested and produced an undersized pool.
    if len(candidates) < top_k and (genre or mood):
        logger.debug(
            f"Candidate pool too small ({len(candidates)}) after soft filters, "
            "expanding to energy-filtered catalog"
        )
        candidates = [
            s for s in catalog if min_energy <= s["energy"] <= max_energy
        ]

    logger.info(
        f"Retriever returned {len(candidates)} candidates "
        f"(genre={genre}, mood={mood}, energy=[{min_energy},{max_energy}])"
    )
    return candidates


def retrieval_summary(candidates: list[dict]) -> str:
    """Return a human-readable summary of retrieved candidates."""
    genres = set(s["genre"] for s in candidates)
    moods = set(s["mood"] for s in candidates)
    avg_energy = sum(s["energy"] for s in candidates) / len(candidates) if candidates else 0
    return (
        f"{len(candidates)} songs retrieved | "
        f"Genres: {', '.join(sorted(genres))} | "
        f"Moods: {', '.join(sorted(moods))} | "
        f"Avg energy: {avg_energy:.2f}"
    )
