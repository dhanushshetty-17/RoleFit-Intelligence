from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import call_claude, log, parse_json_response


class CandidateProfileEnricher:
    def __init__(self, data_dir: str | Path = "data", dry_run: bool = False) -> None:
        self.data_dir = Path(data_dir)
        self.dry_run = dry_run

    def build_profile_text(self, row: Any) -> str:
        record = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        parts: list[str] = []

        def add(label: str, value: Any) -> None:
            if value is None:
                return
            if isinstance(value, float) and pd.isna(value):
                return
            if isinstance(value, str) and not value.strip():
                return
            parts.append(f"{label}: {value}")

        add("candidate_id", record.get("candidate_id"))

        profile = record.get("profile") if isinstance(record.get("profile"), dict) else {}
        if profile:
            add("profile.anonymized_name", profile.get("anonymized_name"))
            add("profile.headline", profile.get("headline"))
            add("profile.summary", profile.get("summary"))
            add("profile.location", profile.get("location"))
            add("profile.country", profile.get("country"))
            add("profile.years_of_experience", profile.get("years_of_experience"))
            add("profile.current_title", profile.get("current_title"))
            add("profile.current_company", profile.get("current_company"))
            add("profile.current_company_size", profile.get("current_company_size"))
            add("profile.current_industry", profile.get("current_industry"))

        career_history = record.get("career_history") or []
        if career_history:
            add("career_history.count", len(career_history))
            for idx, item in enumerate(career_history[:5], start=1):
                if not isinstance(item, dict):
                    continue
                add(f"career_history[{idx}].title", item.get("title"))
                add(f"career_history[{idx}].company", item.get("company"))
                add(f"career_history[{idx}].industry", item.get("industry"))
                add(f"career_history[{idx}].duration_months", item.get("duration_months"))
                add(f"career_history[{idx}].description", item.get("description"))

        education = record.get("education") or []
        if education:
            add("education.count", len(education))
            for idx, item in enumerate(education[:3], start=1):
                if not isinstance(item, dict):
                    continue
                add(f"education[{idx}].institution", item.get("institution"))
                add(f"education[{idx}].degree", item.get("degree"))
                add(f"education[{idx}].field_of_study", item.get("field_of_study"))
                add(f"education[{idx}].tier", item.get("tier"))

        skills = record.get("skills") or []
        if skills:
            add("skills.count", len(skills))
            for idx, item in enumerate(skills[:20], start=1):
                if not isinstance(item, dict):
                    continue
                add(f"skills[{idx}].name", item.get("name"))
                add(f"skills[{idx}].proficiency", item.get("proficiency"))
                add(f"skills[{idx}].endorsements", item.get("endorsements"))
                add(f"skills[{idx}].duration_months", item.get("duration_months"))

        certifications = record.get("certifications") or []
        if certifications:
            add("certifications.count", len(certifications))
            for idx, item in enumerate(certifications[:5], start=1):
                if not isinstance(item, dict):
                    continue
                add(f"certifications[{idx}].name", item.get("name"))
                add(f"certifications[{idx}].issuer", item.get("issuer"))
                add(f"certifications[{idx}].year", item.get("year"))

        languages = record.get("languages") or []
        if languages:
            add("languages.count", len(languages))
            for idx, item in enumerate(languages[:5], start=1):
                if not isinstance(item, dict):
                    continue
                add(f"languages[{idx}].language", item.get("language"))
                add(f"languages[{idx}].proficiency", item.get("proficiency"))

        redrob = record.get("redrob_signals") if isinstance(record.get("redrob_signals"), dict) else {}
        if redrob:
            add("redrob_signals.profile_completeness_score", redrob.get("profile_completeness_score"))
            add("redrob_signals.last_active_date", redrob.get("last_active_date"))
            add("redrob_signals.open_to_work_flag", redrob.get("open_to_work_flag"))
            add("redrob_signals.profile_views_received_30d", redrob.get("profile_views_received_30d"))
            add("redrob_signals.applications_submitted_30d", redrob.get("applications_submitted_30d"))
            add("redrob_signals.recruiter_response_rate", redrob.get("recruiter_response_rate"))
            add("redrob_signals.avg_response_time_hours", redrob.get("avg_response_time_hours"))
            add("redrob_signals.connection_count", redrob.get("connection_count"))
            add("redrob_signals.endorsements_received", redrob.get("endorsements_received"))
            add("redrob_signals.notice_period_days", redrob.get("notice_period_days"))
            add("redrob_signals.preferred_work_mode", redrob.get("preferred_work_mode"))
            add("redrob_signals.willing_to_relocate", redrob.get("willing_to_relocate"))
            add("redrob_signals.github_activity_score", redrob.get("github_activity_score"))
            add("redrob_signals.search_appearance_30d", redrob.get("search_appearance_30d"))
            add("redrob_signals.saved_by_recruiters_30d", redrob.get("saved_by_recruiters_30d"))
            add("redrob_signals.interview_completion_rate", redrob.get("interview_completion_rate"))
            add("redrob_signals.offer_acceptance_rate", redrob.get("offer_acceptance_rate"))
            add("redrob_signals.verified_email", redrob.get("verified_email"))
            add("redrob_signals.verified_phone", redrob.get("verified_phone"))
            add("redrob_signals.linkedin_connected", redrob.get("linkedin_connected"))
            skill_scores = redrob.get("skill_assessment_scores") or {}
            if skill_scores:
                top_scores = sorted(skill_scores.items(), key=lambda item: item[1], reverse=True)[:10]
                add("redrob_signals.skill_assessment_top", top_scores)

        free_text_bits: list[str] = []
        for item in record.get("career_history", []) or []:
            if isinstance(item, dict) and item.get("description"):
                free_text_bits.append(str(item["description"]))
        for item in record.get("skills", []) or []:
            if isinstance(item, dict) and item.get("name"):
                free_text_bits.append(str(item["name"]))
        if free_text_bits:
            parts.append("free_text_signals: " + " | ".join(free_text_bits[:20]))

        return "\n".join(parts)

    def _fallback_enrichment(self, record: dict[str, Any], profile_text: str) -> dict[str, Any]:
        profile = record.get("profile") if isinstance(record.get("profile"), dict) else {}
        skills = [item.get("name") for item in record.get("skills", []) or [] if isinstance(item, dict) and item.get("name")]
        domains = [item.get("industry") for item in record.get("career_history", []) or [] if isinstance(item, dict) and item.get("industry")]
        years = profile.get("years_of_experience") or 0
        return {
            "candidate_id": record.get("candidate_id"),
            "name": profile.get("anonymized_name"),
            "inferred_strengths": skills[:5] or [profile.get("headline") or "Broad experience"],
            "career_trajectory": "pivoting",
            "trajectory_reason": "Fallback heuristic used because API scoring was unavailable.",
            "leadership_evidence": None,
            "initiative_signals": ["Open to work"],
            "red_flags": [],
            "standout_signal": profile.get("summary", "")[:180] or "Profile available for review.",
            "years_experience": int(float(years)) if years is not None else 0,
            "seniority_estimate": "mid",
            "top_skills": skills[:5],
            "domain_history": [item for item in domains if item],
            "profile_text": profile_text,
        }

    async def enrich_candidate(self, row: Any) -> dict[str, Any]:
        record = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        profile_text = self.build_profile_text(record)
        if self.dry_run:
            return self._fallback_enrichment(record, profile_text)

        candidate_id = record.get("candidate_id")
        prompt = f"""
You are a senior recruiter reading a candidate profile. Analyze this profile with the eye of someone who has interviewed thousands of people. Look for real signals, not surface-level keywords.

Candidate Profile:
{profile_text}

Extract these insights as JSON:
{{
"candidate_id": "id from profile or row index",
"name": "candidate name",
"inferred_strengths": ["3-5 genuine strengths backed by evidence in the profile"],
"career_trajectory": "ascending/plateaued/pivoting/inconsistent",
"trajectory_reason": "one sentence explaining why",
"leadership_evidence": "specific evidence of leading people or projects, or null",
"initiative_signals": ["evidence of self-starting: open source, side projects, writing, teaching, etc"],
"red_flags": ["tenure gaps, unexplained drops, skills without evidence — or empty list if none"],
"standout_signal": "the single most impressive thing about this person in one sentence",
"years_experience": estimated number as integer,
"seniority_estimate": "junior/mid/senior/staff",
"top_skills": ["top 5 skills this person actually has based on evidence"],
"domain_history": ["industries or domains they have worked in"]
}}

Respond ONLY with valid JSON.
""".strip()

        try:
            response_text = await call_claude(prompt, max_tokens=1600)
            parsed = parse_json_response(response_text)
            parsed.setdefault("candidate_id", candidate_id)
            parsed.setdefault("name", (record.get("profile") or {}).get("anonymized_name") if isinstance(record.get("profile"), dict) else None)
            parsed["profile_text"] = profile_text
            return parsed
        except Exception as exc:
            log("error", "Candidate enrichment failed", {"candidate_id": candidate_id, "error": str(exc)})
            return self._fallback_enrichment(record, profile_text)

    async def enrich_all(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(8)
        total = len(df)
        completed = 0

        async def run_one(index: int, row: pd.Series) -> dict[str, Any]:
            nonlocal completed
            async with semaphore:
                try:
                    enriched = await self.enrich_candidate(row)
                except Exception as exc:
                    log("error", "Unhandled candidate failure", {"index": index, "error": str(exc)})
                    enriched = self._fallback_enrichment(row.to_dict(), self.build_profile_text(row))
                completed += 1
                if completed % 1000 == 0 or completed == total:
                    log("info", f"Enriched candidate {completed}/{total}", {"candidate_id": enriched.get("candidate_id")})
                return enriched

        results: list[dict[str, Any]] = []
        batch_size = 512
        rows = list(df.iterrows())
        for start in range(0, total, batch_size):
            batch = rows[start : start + batch_size]
            batch_results = await asyncio.gather(*(run_one(original_index, row) for original_index, row in batch))
            results.extend(batch_results)
        return results
