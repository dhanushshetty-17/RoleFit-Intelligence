from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import call_claude, log, parse_json_response


class JobDNAExtractor:
    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.cache_path = self.data_dir / "job_dna.json"

    def load_cached(self) -> dict[str, Any] | None:
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        return None

    def _fallback_extract(self, jd_text: str) -> dict[str, Any]:
        lowered = jd_text.lower()
        if "senior ai engineer" in lowered:
            job_title = "Senior AI Engineer"
        else:
            first_line = next((line.strip() for line in jd_text.splitlines() if line.strip()), "Unknown Role")
            job_title = first_line.replace("Job Description:", "").strip()[:120]

        dealbreakers = []
        for needle in ["production", "embeddings", "retrieval", "ranking", "python", "evaluation", "vector", "hybrid"]:
            if needle in lowered:
                dealbreakers.append(needle)

        core_skills = [
            {"skill": "Python", "weight": 1.0, "is_dealbreaker": True},
            {"skill": "Embeddings", "weight": 0.95, "is_dealbreaker": True},
            {"skill": "Retrieval / Ranking", "weight": 0.95, "is_dealbreaker": True},
            {"skill": "Evaluation", "weight": 0.9, "is_dealbreaker": True},
            {"skill": "LLMs", "weight": 0.8, "is_dealbreaker": False},
            {"skill": "Vector Search", "weight": 0.85, "is_dealbreaker": False},
        ]

        return {
            "job_title": job_title,
            "seniority_level": "senior",
            "core_skills": core_skills,
            "adjacent_skills": ["MLOps", "product engineering", "distributed systems"],
            "domain_expertise": "AI talent intelligence / candidate ranking",
            "org_context": "startup-scaling",
            "leadership_required": True,
            "leadership_evidence": "own the intelligence layer of Redrob's product",
            "hidden_requirements": ["strong product sense", "fast iteration", "works well with ambiguity"],
            "dealbreakers": dealbreakers,
            "what_great_looks_like": "Someone who has shipped retrieval or ranking systems to production and can balance model quality with product velocity. They reason carefully about evaluation and can turn messy signals into reliable ranking behavior.",
            "what_mediocre_looks_like": "Someone who lists many ML keywords but has not shipped a system that changed a business workflow. They may understand tools, but not the operational and product tradeoffs required here.",
        }

    async def extract(self, jd_text: str, use_cache: bool = True, dry_run: bool = False) -> dict[str, Any]:
        if use_cache:
            cached = self.load_cached()
            if cached is not None:
                log("info", "Loaded cached JobDNA", {"path": str(self.cache_path)})
                return cached

        if dry_run:
            job_dna = self._fallback_extract(jd_text)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(job_dna, indent=2, ensure_ascii=False), encoding="utf-8")
            return job_dna

        prompt = f"""
You are an expert talent analyst. Read this job description with deep understanding — not as a keyword scanner, but as someone who truly understands what this role needs.

Extract a JobDNA object as JSON with these exact fields:
{{
"job_title": "the actual role title",
"seniority_level": "junior/mid/senior/staff/lead — inferred from scope",
"core_skills": [{{"skill": "name", "weight": 0.0-1.0, "is_dealbreaker": true/false}}],
"adjacent_skills": ["skills that suggest high potential even if not required"],
"domain_expertise": "specific industry or technical domain needed",
"org_context": "startup-scaling/enterprise/research/product — from language register",
"leadership_required": true/false,
"leadership_evidence": "quote from JD that signals this or null",
"hidden_requirements": ["things implied but not stated — decode the subtext"],
"dealbreakers": ["hard disqualifiers explicit or strongly implied"],
"what_great_looks_like": "in 2 sentences, describe a truly great candidate for this role",
"what_mediocre_looks_like": "in 2 sentences, describe someone who looks ok on paper but isnt right"
}}

Job Description:
{jd_text}

Respond ONLY with valid JSON. No markdown, no explanation, just the JSON object.
""".strip()

        log("step", "Extracting JobDNA from JD")
        response_text = await call_claude(prompt, max_tokens=1800)
        job_dna = parse_json_response(response_text)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(job_dna, indent=2, ensure_ascii=False), encoding="utf-8")
        return job_dna
