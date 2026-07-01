from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


class RankedOutputGenerator:
    def __init__(self, output_dir: str | Path = "output") -> None:
        self.output_dir = Path(output_dir)

    def generate(self, ranked_results: list[dict[str, Any]], output_format_hint: dict[str, Any] | None = None) -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ranked_path = self.output_dir / "ranked_candidates.csv"
        full_path = self.output_dir / "full_analysis.json"
        breakdown_path = self.output_dir / "score_breakdown.csv"
        briefs_path = self.output_dir / "candidate_briefs.csv"

        header = ["candidate_id", "rank", "score", "reasoning"]
        if output_format_hint and isinstance(output_format_hint, dict):
            hint_columns = output_format_hint.get("columns") or output_format_hint.get("header")
            if isinstance(hint_columns, list) and hint_columns:
                header = list(hint_columns)

        with ranked_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for result in sorted(ranked_results, key=lambda item: item["rank"]):
                row = {
                    "candidate_id": result.get("candidate_id", ""),
                    "rank": result.get("rank", ""),
                    "score": f"{result.get('final_score', 0.0):.4f}",
                    "reasoning": result.get("reason_for_fit", ""),
                }
                row.update({key: result.get(key, "") for key in header if key not in row})
                writer.writerow({field: row.get(field, "") for field in header})

        with full_path.open("w", encoding="utf-8") as f:
            json.dump(ranked_results, f, indent=2, ensure_ascii=False, default=str)

        breakdown_fields = ["rank", "candidate_id", "name", "semantic_score", "hard_score", "llm_score", "trajectory_bonus", "final_score"]
        with breakdown_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=breakdown_fields)
            writer.writeheader()
            for result in sorted(ranked_results, key=lambda item: item["rank"]):
                writer.writerow({field: result.get(field, "") for field in breakdown_fields})

        brief_fields = [
            "rank",
            "candidate_id",
            "name",
            "final_score",
            "fit_verdict",
            "reason_for_fit",
            "biggest_strength",
            "biggest_risk",
            "standout_signal",
            "red_flags",
            "interview_probes",
            "semantic_score",
            "hard_score",
            "llm_score",
            "trajectory_bonus",
        ]
        with briefs_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=brief_fields)
            writer.writeheader()
            for result in sorted(ranked_results, key=lambda item: item["rank"]):
                row = {field: result.get(field, "") for field in brief_fields}
                row["red_flags"] = json.dumps(result.get("red_flags", []), ensure_ascii=False, default=str)
                row["interview_probes"] = json.dumps(result.get("interview_probes", []), ensure_ascii=False, default=str)
                writer.writerow(row)

        self._print_summary(ranked_results[:10])
        self._print_distribution(ranked_results)
        return {"ranked": ranked_path, "full": full_path, "breakdown": breakdown_path, "briefs": briefs_path}

    def _print_summary(self, top_results: list[dict[str, Any]]) -> None:
        print("\nTop 10 Candidates")
        print("=" * 92)
        print(f"{'Rank':<6}{'Name':<28}{'Score':<10}{'Verdict':<18}")
        print("-" * 92)
        for item in top_results:
            print(f"{item.get('rank', ''):<6}{str(item.get('name', ''))[:26]:<28}{item.get('final_score', 0):<10.4f}{str(item.get('fit_verdict', '')):<18}")

    def _print_distribution(self, ranked_results: list[dict[str, Any]]) -> None:
        scores = [float(item.get("final_score", 0.0)) for item in ranked_results]
        verdicts: dict[str, int] = {}
        for item in ranked_results:
            verdict = str(item.get("fit_verdict", "unknown"))
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
        print("\nScore Distribution")
        print("=" * 92)
        print(f"mean: {mean(scores):.4f} | median: {median(scores):.4f} | std: {pstdev(scores):.4f}")
        print("verdict counts:")
        for verdict, count in sorted(verdicts.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {verdict}: {count}")
