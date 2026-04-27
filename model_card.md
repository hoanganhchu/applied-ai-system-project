# Model Card — Applied AI Music Recommender

## Model Overview

**System Name:** Applied AI Music Recommender  
**Version:** 2.0 (Final Project Extension)  
**Base Project:** ai110-musicrecommendersimulation-starter (Module 3)  
**Type:** Rule-based content-based recommender with RAG retrieval and agentic workflow  
**Algorithm:** Weighted attribute scoring (genre, mood, energy, valence) + Plan/Act/Reflect agent loop

---

## What This System Does

Given a user profile (preferred genre, mood, target energy, target valence), the system:
1. Retrieves a relevant subset of songs from the catalog (RAG)
2. Scores each candidate song using weighted attribute matching
3. Reflects on output quality and retries with relaxed filters if needed
4. Returns top-K recommendations with scores, confidence ratings, and explanations

---

## Intended Use

- Educational demonstration of content-based filtering and RAG architecture
- Portfolio project showing AI system design skills
- Learning tool for understanding how recommendation algorithms make decisions

**Not intended for:** production music streaming, commercial use, or any system where recommendation quality has significant user impact.

---

## Limitations

**Dataset size:** Only 25 songs. Real systems use millions of tracks. With 25 songs, the top recommendations can feel repetitive, especially for niche genres with only 2-3 songs.

**No collaborative filtering:** The system ignores what other users with similar tastes liked. Real platforms like Spotify weight collaborative signals heavily, which this system cannot do.

**Feature simplicity:** The system uses only 4 attributes (genre, mood, energy, valence). It ignores lyrics, language, cultural context, artist novelty, listening history, time of day, and context signals.

**Exact genre matching:** Genre matching is exact string comparison. "pop" and "kpop" are treated as completely different, even though a kpop fan might enjoy pop songs. Hierarchical genre taxonomies would be more accurate.

**Static weights:** All users share the same default weights. Real personalization would learn weights per user over time.

**Filter bubble risk:** Repeatedly recommending the same top-genre songs reduces diversity. There is no diversity injection or novelty bonus in the current scoring.

---

## Bias Risks

**Genre over-representation:** If genre_weight is set high (default: 2.0), songs from the preferred genre will dominate the top results even if their energy/valence is a poor match. This can create a filter bubble.

**Mood label subjectivity:** Mood labels ("happy", "intense", "peaceful") are manually assigned to songs. Different people assign different moods to the same song — a song labeled "peaceful" might feel melancholic to someone else.

**Cultural bias:** The catalog is primarily Western genres (pop, rock, EDM). Non-Western genres (afrobeats, kpop, reggae) have fewer songs, so users with those preferences receive fewer genre-matched results and lower confidence scores.

---

## Could This System Be Misused?

**Catalog manipulation:** If someone controls the catalog (adds songs with favorable attributes), they could make specific songs appear at the top of recommendations for all users. Mitigation: catalog curation should be audited.

**Profile targeting:** Constructing adversarial profiles that always surface specific songs is trivially possible with this transparent scoring system. Mitigation: not a concern for educational use; production systems should obscure weights.

**False confidence:** High confidence scores could mislead users into trusting recommendations that are based on a 25-song catalog. Mitigation: always display the catalog size alongside confidence.

---

## Testing Results

| Test | Profile | Avg Confidence | Passed |
|------|---------|----------------|--------|
| TC01 | High-Energy Pop | 0.824 | ✅ |
| TC02 | Chill Lofi | 0.724 | ✅ |
| TC03 | Intense Rock | 0.755 | ✅ |
| TC04 | Peaceful Classical | 0.812 | ✅ |
| TC05 | Festival EDM | 0.796 | ✅ |
| TC06 | Kpop Happy | 0.816 | ✅ |
| TC07 | Unknown genre (jazz) | 0.544 | ✅ |
| TC08 | Extreme energy metal | 0.675 | ✅ |

**8/8 tests passed. Average confidence: 0.743.**

What surprised me: the unknown genre (jazz) test still passed because the fallback mechanism gracefully expanded the candidate pool and the confidence score honestly reflected the weaker match (0.544 vs 0.8+). The system "knew" it was uncertain.

---

## AI Collaboration Notes

This project was developed with AI assistance (Claude). Below are two specific examples:

### Helpful AI Suggestion
When designing the retriever, the AI suggested implementing **soft filters with automatic fallback** instead of hard genre/mood filters. The original plan was to filter strictly by genre first and throw an error if too few results were found. The AI suggested checking if the filtered pool was large enough and falling back to the full catalog if not. This was the right call — it's exactly what production RAG systems do to ensure the retrieval stage never bottlenecks the pipeline.

### Flawed AI Suggestion
The AI initially suggested using **cosine similarity on a feature vector** for retrieval (treating genre/mood as one-hot encoded dimensions alongside energy/valence). This sounded sophisticated but was actually wrong for this use case — cosine similarity on a mixed categorical+continuous vector would give misleading "similarity" scores because the scale differences between binary (0/1) genre encodings and continuous energy values (0.0-1.0) make the geometry meaningless. The manual weighted scoring approach is more transparent and produces more interpretable results for this scale of problem.

---

## What This Project Taught Me

Building this system made two things very clear:

1. **Architecture matters more than algorithms.** The single biggest quality improvement came from separating the retriever from the scorer — not from tuning weights. Clean separation of concerns made debugging, testing, and explaining the system dramatically easier.

2. **Transparency is a feature.** The reason attribute — showing *why* each song scored the way it did — is more valuable than the score itself. A score of 6.92 is meaningless; "genre match + mood match + perfect energy alignment" is actionable.

The agentic loop (Plan/Act/Reflect) reinforced that AI systems benefit from self-evaluation. Building a system that *checks its own work* and retries when uncertain is a habit that applies far beyond music recommendation.
