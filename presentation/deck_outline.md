# Talent Intelligence System — Presentation Deck

## Slide 1 — Title
Talent Intelligence System
AI-Powered Candidate Ranking, Explained
From job description to ranked shortlist in minutes.

## Slide 2 — The Problem
Hiring does not scale when candidate volume grows.
Keyword search over-ranks noise, embeddings miss hard constraints, and manual review collapses at scale.
Recruiters need ranked recommendations with reasons they can trust.

## Slide 3 — What I Built
A hybrid candidate ranking pipeline that combines:
- JobDNA extraction from the job description
- Candidate profile enrichment into recruiter-friendly signals
- Semantic similarity for broad relevance
- Hard-constraint matching for must-have skills
- LLM judgment for top candidates only
- Trajectory and red-flag signals for quality control

## Slide 4 — Why This Design
No single signal is trusted alone.
Semantic similarity finds relevant profiles, BM25 enforces dealbreakers, the LLM adds human-like judgment for the top set, and the trajectory layer rewards upward momentum while penalizing weak signals.
This keeps the system explainable, affordable, and scalable.

## Slide 5 — How It Works
1. Ingest the job description and candidate dataset.
2. Extract a structured JobDNA object.
3. Build enriched candidate profiles.
4. Score candidates with four weighted signals.
5. Run LLM review only on top-N candidates.
6. Export ranked candidates and recruiter briefs.

## Slide 6 — Outputs and UI
The system produces:
- ranked_candidates.csv for the shortlist
- score_breakdown.csv for signal-level transparency
- candidate_briefs.csv for quick review
- full_analysis.json for deep inspection

The Streamlit dashboard adds search, filters, compare mode, and candidate drill-down.

## Slide 7 — Example Impact
The pipeline ranked 100,000 candidates end to end.
It runs in dry-run mode with deterministic fallbacks and in live mode with Anthropic API scoring.
That makes it useful for local testing, demoing, and production-style evaluation.

## Slide 8 — Roadmap
Next steps:
- Calibrate weights against recruiter outcomes
- Add role-specific weight profiles
- Build offline regression benchmarks
- Extend trajectory analysis across more signals
- Cache repeated JD runs for lower cost
