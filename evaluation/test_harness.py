"""
test_harness.py — Automated evaluation script for the Applied AI Music Recommender.

Runs predefined test cases and prints a pass/fail summary with confidence
ratings. Satisfies the "Test Harness or Evaluation Script" stretch feature.

Run with:
    python -m evaluation.test_harness
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent import run_agent
from src.retriever import load_catalog
from src.scorer import UserProfile

# Shared catalog (loaded once)
CATALOG = load_catalog()


# ── Test case definitions ────────────────────────────────────────────────────
# Each case: (label, profile, expected_genre_in_top1, min_confidence)

TEST_CASES = [
    {
        "id": "TC01",
        "label": "High-Energy Pop Profile",
        "profile": UserProfile(genre="pop", mood="happy", target_energy=0.85, target_valence=0.85),
        "expect_genre_in_top1": "pop",
        "min_avg_confidence": 0.50,
    },
    {
        "id": "TC02",
        "label": "Chill Lofi Profile",
        "profile": UserProfile(genre="lofi", mood="chill", target_energy=0.28, target_valence=0.40),
        "expect_genre_in_top1": "lofi",
        "min_avg_confidence": 0.50,
    },
    {
        "id": "TC03",
        "label": "Intense Rock Profile",
        "profile": UserProfile(genre="rock", mood="intense", target_energy=0.90, target_valence=0.30),
        "expect_genre_in_top1": "rock",
        "min_avg_confidence": 0.50,
    },
    {
        "id": "TC04",
        "label": "Peaceful Classical Profile",
        "profile": UserProfile(genre="classical", mood="peaceful", target_energy=0.18, target_valence=0.65),
        "expect_genre_in_top1": "classical",
        "min_avg_confidence": 0.50,
    },
    {
        "id": "TC05",
        "label": "EDM Festival Profile",
        "profile": UserProfile(genre="edm", mood="energetic", target_energy=0.93, target_valence=0.55),
        "expect_genre_in_top1": "edm",
        "min_avg_confidence": 0.50,
    },
    {
        "id": "TC06",
        "label": "Kpop Happy Profile",
        "profile": UserProfile(genre="kpop", mood="happy", target_energy=0.82, target_valence=0.88),
        "expect_genre_in_top1": "kpop",
        "min_avg_confidence": 0.50,
    },
    {
        "id": "TC07",
        "label": "Edge Case: Unknown genre (jazz)",
        "profile": UserProfile(genre="jazz", mood="chill", target_energy=0.40, target_valence=0.55),
        "expect_genre_in_top1": None,  # No jazz in catalog — just check it doesn't crash
        "min_avg_confidence": 0.30,   # Lower bar for fallback
    },
    {
        "id": "TC08",
        "label": "Edge Case: Extreme energy (0.99) metal",
        "profile": UserProfile(genre="metal", mood="intense", target_energy=0.99, target_valence=0.20),
        "expect_genre_in_top1": "metal",
        "min_avg_confidence": 0.45,
    },
]


def run_test(case: dict) -> dict:
    """Run one test case and return a result dict."""
    try:
        result = run_agent(case["profile"], top_k=5, catalog=CATALOG)
        top1_genre = result.recommendations[0].song["genre"] if result.recommendations else None
        avg_conf = result.avg_confidence

        genre_pass = (
            case["expect_genre_in_top1"] is None
            or top1_genre == case["expect_genre_in_top1"]
        )
        conf_pass = avg_conf >= case["min_avg_confidence"]
        passed = genre_pass and conf_pass

        return {
            "id": case["id"],
            "label": case["label"],
            "passed": passed,
            "genre_pass": genre_pass,
            "conf_pass": conf_pass,
            "top1_genre": top1_genre,
            "expected_genre": case["expect_genre_in_top1"],
            "avg_confidence": avg_conf,
            "min_confidence": case["min_avg_confidence"],
            "top1_title": result.recommendations[0].song["title"] if result.recommendations else "N/A",
            "retry_count": result.retry_count,
            "error": None,
        }
    except Exception as e:
        return {
            "id": case["id"],
            "label": case["label"],
            "passed": False,
            "error": str(e),
        }


def print_summary(results: list[dict]):
    """Print a formatted test summary table."""
    sep = "─" * 75
    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print("\n" + "═" * 75)
    print("  🧪  TEST HARNESS SUMMARY")
    print("═" * 75)
    print(f"  {'ID':<6} {'Label':<35} {'Status':<8} {'Top-1 Genre':<12} {'Avg Conf'}")
    print(sep)

    for r in results:
        if r.get("error"):
            print(f"  {r['id']:<6} {r['label']:<35} {'ERROR':<8} {'N/A':<12} {r['error'][:20]}")
        else:
            status = "✅ PASS" if r["passed"] else "❌ FAIL"
            genre_str = r.get("top1_genre", "N/A") or "N/A"
            print(
                f"  {r['id']:<6} {r['label']:<35} {status:<8} "
                f"{genre_str:<12} {r.get('avg_confidence', 0):.3f}"
            )

    print(sep)
    print(f"\n  Result: {passed}/{total} tests passed")

    # Confidence stats
    confs = [r["avg_confidence"] for r in results if "avg_confidence" in r]
    if confs:
        print(f"  Avg confidence across all tests: {sum(confs)/len(confs):.3f}")
        print(f"  Min confidence: {min(confs):.3f} | Max: {max(confs):.3f}")

    # Failures detail
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n  ⚠️  Failures ({len(failures)}):")
        for f in failures:
            if f.get("error"):
                print(f"    [{f['id']}] Exception: {f['error']}")
            else:
                if not f.get("genre_pass"):
                    print(
                        f"    [{f['id']}] Genre mismatch: "
                        f"expected '{f['expected_genre']}', got '{f['top1_genre']}'"
                    )
                if not f.get("conf_pass"):
                    print(
                        f"    [{f['id']}] Confidence too low: "
                        f"{f['avg_confidence']:.3f} < {f['min_confidence']}"
                    )

    print("\n" + "═" * 75 + "\n")


def main():
    print("\n🎵 Applied AI Music Recommender — Test Harness")
    print(f"Running {len(TEST_CASES)} test cases...\n")

    results = []
    for case in TEST_CASES:
        print(f"  [{case['id']}] {case['label']} ...", end=" ", flush=True)
        r = run_test(case)
        results.append(r)
        print("✅" if r["passed"] else "❌")

    print_summary(results)

    # Exit code: 0 if all passed, 1 otherwise
    all_passed = all(r["passed"] for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
