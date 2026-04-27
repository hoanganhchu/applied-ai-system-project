import argparse
import logging
import sys

from .agent import run_agent
from .scorer import UserProfile

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("recommender.log", mode="a", encoding="utf-8"),
    ],
)
logging.getLogger().handlers[0].setLevel(logging.WARNING)
logging.getLogger().handlers[1].setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# ── Predefined demo profiles ────────────────────────────────────────────────
DEMO_PROFILES = {
    "1": UserProfile(
        genre="pop", mood="happy",
        target_energy=0.85, target_valence=0.85,
        genre_weight=2.0, mood_weight=1.5,
        energy_weight=2.0, valence_weight=1.5,
    ),
    "2": UserProfile(
        genre="lofi", mood="chill",
        target_energy=0.28, target_valence=0.40,
        genre_weight=2.0, mood_weight=1.5,
        energy_weight=2.0, valence_weight=1.5,
    ),
    "3": UserProfile(
        genre="rock", mood="intense",
        target_energy=0.90, target_valence=0.30,
        genre_weight=2.0, mood_weight=1.5,
        energy_weight=2.5, valence_weight=1.0,
    ),
    "4": UserProfile(
        genre="classical", mood="peaceful",
        target_energy=0.18, target_valence=0.65,
        genre_weight=2.0, mood_weight=1.5,
        energy_weight=2.0, valence_weight=1.5,
    ),
    "5": UserProfile(
        genre="edm", mood="energetic",
        target_energy=0.93, target_valence=0.55,
        genre_weight=2.0, mood_weight=1.5,
        energy_weight=2.5, valence_weight=1.0,
    ),
}

PROFILE_LABELS = {
    "1": "High-Energy Pop",
    "2": "Chill Lofi",
    "3": "Deep Intense Rock",
    "4": "Peaceful Classical",
    "5": "Festival EDM",
}

DEFAULT_PROFILE_KEY = "1"

ALLOWED_GENRES = {"pop", "lofi", "rock", "classical", "edm", "metal", "folk", "kpop", "country", "reggae", "afrobeats"}
ALLOWED_MOODS = {"happy", "chill", "intense", "peaceful", "energetic", "nostalgic", "melancholic", "romantic"}


def print_separator(char="─", width=60):
    print(char * width)


def display_result(result, profile_label: str, show_steps: bool = True):
    """Pretty-print agent result to stdout."""
    print(f"\n{profile_label}")
    print_separator()

    if show_steps:
        print("Agent reasoning steps:")
        for step in result.steps:
            print(f"- {step.step_name}: {step.description}")

    print(
        f"Retries: {result.retry_count} | "
        f"Avg confidence: {result.avg_confidence:.3f} | "
        f"Quality: {'PASSED' if result.passed_quality_check else 'LOW'}"
    )

    print_separator()
    print(f"Top {len(result.recommendations)} recommendations:")

    for i, rec in enumerate(result.recommendations, 1):
        s = rec.song
        print(
            f"{i}. {s['title']} — {s['artist']}"
            f" | score={rec.score:.3f}"
            f" | confidence={rec.confidence:.3f}"
            f" | genre={s['genre']}"
            f" | mood={s['mood']}"
        )

    print_separator()


def run_demo():
    """Run all demo profiles automatically."""
    logger.info("Starting demo run for all profiles")
    for key, label in PROFILE_LABELS.items():
        profile = DEMO_PROFILES[key]
        result = run_agent(profile, top_k=5)
        display_result(result, label, show_steps=True)
    print("\nDemo complete. See recommender.log for full trace.\n")


def run_default_demo():
    """Run one concise demo profile by default."""
    logger.info("Starting default demo run")
    profile = DEMO_PROFILES[DEFAULT_PROFILE_KEY]
    result = run_agent(profile, top_k=5)
    display_result(result, PROFILE_LABELS[DEFAULT_PROFILE_KEY], show_steps=False)
    print("\nDone. Use --demo to run all presets or --interactive for custom input.\n")


def _prompt_choice(prompt_text: str, allowed_values: set[str], default_value: str) -> str:
    """Prompt for a value and fall back to the default on blank or invalid input."""
    raw_value = input(prompt_text).strip().lower()
    if not raw_value:
        return default_value
    if raw_value in allowed_values:
        return raw_value
    print(f"Invalid input, using default: {default_value}")
    return default_value


def _prompt_float(prompt_text: str, default_value: float) -> float:
    """Prompt for a float in [0, 1] and fall back to the default on invalid input."""
    raw_value = input(prompt_text).strip()
    if not raw_value:
        return default_value
    try:
        value = float(raw_value)
    except ValueError:
        print(f"Invalid input, using default: {default_value}")
        return default_value
    return max(0.0, min(1.0, value))


def run_interactive():
    """Interactive CLI mode for custom profiles."""
    print("\nCustom profile builder\n")

    genre = _prompt_choice(
        "Preferred genre (press Enter for pop): ",
        ALLOWED_GENRES,
        "pop",
    )
    mood = _prompt_choice(
        "Preferred mood (press Enter for happy): ",
        ALLOWED_MOODS,
        "happy",
    )
    energy = _prompt_float("Target energy (0.0 = calm, 1.0 = high; Enter for 0.7): ", 0.7)
    valence = _prompt_float("Target valence (0.0 = dark, 1.0 = upbeat; Enter for 0.7): ", 0.7)

    profile = UserProfile(
        genre=genre, mood=mood,
        target_energy=max(0.0, min(1.0, energy)),
        target_valence=max(0.0, min(1.0, valence)),
    )

    result = run_agent(profile, top_k=5)
    display_result(result, f"Custom: {genre.capitalize()} / {mood.capitalize()}", show_steps=True)


def main():
    parser = argparse.ArgumentParser(
        description="Applied AI Music Recommender"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run all preset demo profiles",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="build a custom profile interactively",
    )
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  Applied AI Music Recommender")
    print("═" * 60)

    if args.interactive:
        run_interactive()
    elif args.demo:
        run_demo()
    else:
        run_default_demo()


if __name__ == "__main__":
    main()
