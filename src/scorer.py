from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import call_claude, log, normalize_score, parse_json_response

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - import fallback for constrained environments
    SentenceTransformer = None  # type: ignore[assignment]


class HybridScoringEngine:
    def __init__(self, job_dna: dict[str, Any], top_n_llm: int = 40, dry_run: bool = False) -> None:
        self.job_dna = job_dna
        self.top_n_llm = top_n_llm
        self.dry_run = dry_run
        self.model = None
        self.vectorizer = HashingVectorizer(n_features=1024, alternate_sign=False, norm="l2")
        self.job_text = self._job_text(job_dna)
        self.job_embedding = self._encode_texts([self.job_text])

        if SentenceTransformer is not None and not self.dry_run:
            try:
                self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
                self.job_embedding = self.model.encode([self.job_text], normalize_embeddings=True)
            except Exception as exc:
                log("warn", "SentenceTransformer unavailable, using hashing fallback", {"error": str(exc)})

    def _job_text(self, job_dna: dict[str, Any]) -> str:
        fields = [
            job_dna.get("job_title", ""),
            job_dna.get("seniority_level", ""),
            job_dna.get("domain_expertise", ""),
            job_dna.get("org_context", ""),
            job_dna.get("what_great_looks_like", ""),
            job_dna.get("what_mediocre_looks_like", ""),
        ]
        fields.extend(skill.get("skill", "") for skill in (job_dna.get("core_skills", []) or []) if isinstance(skill, dict))
        fields.extend(job_dna.get("adjacent_skills", []) or [])
        fields.extend(job_dna.get("hidden_requirements", []) or [])
        fields.extend(job_dna.get("dealbreakers", []) or [])
        return "\n".join(str(field) for field in fields if field)

    def _candidate_text(self, enriched: dict[str, Any]) -> str:
        return str(enriched.get("profile_text", ""))

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if self.model is not None:
            return np.asarray(self.model.encode(texts, normalize_embeddings=True))
        return self.vectorizer.transform(texts).toarray()

    def _semantic_scores(self, enriched_profiles: list[dict[str, Any]]) -> np.ndarray:
        texts = [self._candidate_text(profile) for profile in enriched_profiles]
        candidate_embeddings = self._encode_texts(texts)
        scores = cosine_similarity(candidate_embeddings, self.job_embedding).reshape(-1)
        return np.clip((scores + 1.0) / 2.0, 0.0, 1.0)

    def _tokenize(self, text: str) -> list[str]:
        return [token for token in text.lower().replace("/", " ").replace("-", " ").split() if token]

    def _hard_scores(self, enriched_profiles: list[dict[str, Any]]) -> np.ndarray:
        corpus_texts = [self._candidate_text(profile) for profile in enriched_profiles]
        tokenized_corpus = [self._tokenize(text) for text in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        dealbreakers = [
            skill.get("skill", "")
            for skill in (self.job_dna.get("core_skills", []) or [])
            if isinstance(skill, dict) and skill.get("is_dealbreaker")
        ]
        query_tokens = self._tokenize(" ".join(dealbreakers + self._tokenize(self.job_dna.get("job_title", ""))))
        if not query_tokens:
            query_tokens = self._tokenize(self.job_text)

        raw_scores = np.asarray(bm25.get_scores(query_tokens), dtype=float)
        raw_min = float(raw_scores.min()) if len(raw_scores) else 0.0
        raw_max = float(raw_scores.max()) if len(raw_scores) else 1.0
        norm_scores = np.asarray([normalize_score(score, raw_min, raw_max) for score in raw_scores], dtype=float)

        dealbreaker_tokens = {token for text in dealbreakers for token in self._tokenize(text)}
        for idx, profile in enumerate(enriched_profiles):
            text_tokens = set(self._tokenize(corpus_texts[idx]))
            missed = len(dealbreaker_tokens - text_tokens)
            if missed > 2:
                norm_scores[idx] = max(0.0, norm_scores[idx] - 0.3)
        return norm_scores

    def _trajectory_bonus(self, enriched: dict[str, Any]) -> float:
        bonus = 0.0
        trajectory = str(enriched.get("career_trajectory", "")).lower()
        if trajectory == "ascending":
            bonus += 0.05
        if len(enriched.get("initiative_signals", []) or []) >= 2:
            bonus += 0.04
        if len(enriched.get("red_flags", []) or []) == 0:
            bonus += 0.03
        if trajectory == "inconsistent":
            bonus -= 0.05
        if len(enriched.get("red_flags", []) or []) >= 3:
            bonus -= 0.05

        years = float(enriched.get("years_experience") or 0)
        implied = {"junior": 2, "mid": 5, "senior": 8, "staff": 10, "lead": 9}.get(str(self.job_dna.get("seniority_level", "mid")).lower(), 5)
        if years < implied - 2:
            bonus -= 0.03
        return max(-0.10, min(0.10, bonus))

    async def _llm_score_candidate(self, enriched: dict[str, Any]) -> dict[str, Any]:
        if self.dry_run:
            return {
                "skill_match": 5,
                "skill_match_reason": "Dry run mode.",
                "experience_depth": 5,
                "experience_depth_reason": "Dry run mode.",
                "trajectory_fit": 5,
                "trajectory_fit_reason": "Dry run mode.",
                "culture_signal": 5,
                "culture_signal_reason": "Dry run mode.",
                "overall_llm_score": 5,
                "fit_verdict": "not_evaluated",
                "reason_for_fit": "Dry run mode.",
                "biggest_strength": enriched.get("standout_signal", ""),
                "biggest_risk": "Dry run mode.",
                "interview_probes": [],
            }

        prompt = f"""
You are evaluating a candidate for a specific role. Be honest and precise — a score of 7 means genuinely strong, not just okay. Reserve 9-10 for exceptional matches.

JOB DNA:
{self.job_dna}

CANDIDATE ENRICHED PROFILE:
{enriched}

Score this candidate from 0-10 on each dimension:
{{
"skill_match": 0-10,
"skill_match_reason": "one sentence",
"experience_depth": 0-10,
"experience_depth_reason": "one sentence",
"trajectory_fit": 0-10,
"trajectory_fit_reason": "one sentence",
"culture_signal": 0-10,
"culture_signal_reason": "one sentence",
"overall_llm_score": 0-10,
"fit_verdict": "strong_fit/good_fit/stretch_fit/poor_fit",
"reason_for_fit": "exactly 2 sentences a recruiter can paste directly into a briefing. Be specific about this candidate, not generic.",
"biggest_strength": "one sentence",
"biggest_risk": "one sentence — what would make you hesitate?",
"interview_probes": ["2 specific questions to validate the biggest uncertainties about this candidate"]
}}

Respond ONLY with valid JSON.
""".strip()
        text = await call_claude(prompt, max_tokens=1600)
        return parse_json_response(text)

    async def score_all(self, enriched_profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not enriched_profiles:
            return []

        semantic_scores = self._semantic_scores(enriched_profiles)
        hard_scores = self._hard_scores(enriched_profiles)
        combined_pre_llm = 0.5 * semantic_scores + 0.5 * hard_scores
        top_indices = set(np.argsort(-combined_pre_llm)[: min(self.top_n_llm, len(enriched_profiles))].tolist())

        llm_results: dict[int, dict[str, Any]] = {}
        semaphore = asyncio.Semaphore(5)

        async def run_one(index: int) -> None:
            async with semaphore:
                try:
                    llm_results[index] = await self._llm_score_candidate(enriched_profiles[index])
                except Exception as exc:
                    log("error", "LLM scoring failed", {"candidate_id": enriched_profiles[index].get("candidate_id"), "error": str(exc)})
                    llm_results[index] = {
                        "overall_llm_score": None,
                        "fit_verdict": "not_evaluated",
                        "reason_for_fit": "LLM evaluation failed.",
                        "biggest_strength": enriched_profiles[index].get("standout_signal", ""),
                        "biggest_risk": "; ".join(enriched_profiles[index].get("red_flags", []) or []) or "Unknown",
                        "interview_probes": [],
                    }

        await asyncio.gather(*(run_one(index) for index in sorted(top_indices)))

        results: list[dict[str, Any]] = []
        for idx, enriched in enumerate(enriched_profiles):
            llm = llm_results.get(idx, {})
            llm_score = llm.get("overall_llm_score")
            llm_component = (float(llm_score) / 10.0) if llm_score is not None else float(combined_pre_llm[idx])
            trajectory_bonus = self._trajectory_bonus(enriched)
            final_score = (
                0.30 * float(semantic_scores[idx])
                + 0.25 * float(hard_scores[idx])
                + 0.35 * llm_component
                + 0.10 * trajectory_bonus
            )
            final_score = max(0.0, min(1.0, final_score))
            results.append(
                {
                    "candidate_id": enriched.get("candidate_id"),
                    "name": enriched.get("name"),
                    "final_score": final_score,
                    "semantic_score": float(semantic_scores[idx]),
                    "hard_score": float(hard_scores[idx]),
                    "llm_score": llm_score,
                    "trajectory_bonus": trajectory_bonus,
                    "fit_verdict": llm.get("fit_verdict", "not_evaluated"),
                    "reason_for_fit": llm.get("reason_for_fit", ""),
                    "biggest_strength": llm.get("biggest_strength", enriched.get("standout_signal", "")),
                    "biggest_risk": llm.get("biggest_risk", "; ".join(enriched.get("red_flags", []) or []) or ""),
                    "interview_probes": llm.get("interview_probes", []),
                    "red_flags": enriched.get("red_flags", []),
                    "standout_signal": enriched.get("standout_signal", ""),
                    "profile_text": enriched.get("profile_text", ""),
                }
            )

        results.sort(key=lambda item: (-item["final_score"], item["candidate_id"]))
        for rank, result in enumerate(results, start=1):
            result["rank"] = rank
        return results
