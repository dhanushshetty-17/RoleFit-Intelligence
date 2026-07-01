#!/usr/bin/env python3
"""
Talent Intelligence System — India Runs Data & AI Hackathon
Run: python run.py
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.jd_parser import JobDNAExtractor
from src.output import RankedOutputGenerator
from src.profile_builder import CandidateProfileEnricher
from src.scorer import HybridScoringEngine
from src.utils import load_env, log


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def banner() -> None:
    print(r"""
 ______        _             _     _       _              _____ _       _                _
|_   _ \      (_)           (_)   (_)     (_)            /  __ \ |     | |              | |
  | |_) | __ _ _ _ __  _ __  _ ___ _  __ _ _ _ __   __ _| /  \/ | ___ | |__   ___  __ _| |_ ___
  |  _ < / _` | | '_ \| '_ \| / __| |/ _` | | '_ \ / _` | |   | |/ _ \| '_ \ / _ \/ _` | __/ __|
  | |_) | (_| | | | | | | | | \__ \ | (_| | | | | | (_| | \__/\ | (_) | |_) |  __/ (_| | |_\__ \
  |____/ \__,_|_|_| |_|_| |_|_|___/_|\__, |_|_| |_|\__,_|\____/_|\___/|_.__/ \___|\__,_|\__|___/
                                      __/ |
                                     |___/
""")


def _read_docx_text(path: Path) -> str:
    import xml.etree.ElementTree as ET
    import zipfile

    with zipfile.ZipFile(path) as zf:
        xml_data = zf.read("word/document.xml")
    root = ET.fromstring(xml_data)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:body/w:p", namespace):
        texts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
        if texts:
            paragraphs.append("".join(texts))
    return "\n".join(paragraphs)


def detect_jd_file(data_dir: Path) -> Path:
    matches = [
        path for path in sorted(data_dir.glob("**/*"))
        if path.is_file() and ("jd" in path.name.lower() or "job" in path.name.lower() or "description" in path.name.lower() or path.suffix.lower() == ".txt")
    ]
    if not matches:
        raise FileNotFoundError("Could not auto-detect a job description file in data/.")
    return matches[0]


def detect_candidate_file(data_dir: Path) -> Path:
    matches = [
        path for path in sorted(data_dir.glob("**/*"))
        if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".json", ".jsonl", ".gz"}
        and "sample_submission" not in path.name.lower()
        and "schema" not in path.name.lower()
        and "metadata" not in path.name.lower()
    ]
    if not matches:
        raise FileNotFoundError("Could not auto-detect a candidate data file in data/.")
    return matches[0]


def load_candidates(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(data if isinstance(data, list) else [data])
    if suffix == ".jsonl" or path.name.lower().endswith(".jsonl.gz"):
        opener = gzip.open if path.name.lower().endswith(".gz") else open
        records: list[dict[str, Any]] = []
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)
    raise ValueError(f"Unsupported candidate file format: {path}")


def load_jd_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return _read_docx_text(path)
    return path.read_text(encoding="utf-8")


def load_output_format_hint(data_dir: Path) -> dict[str, Any] | None:
    sample = data_dir / "sample_submission.csv"
    if sample.exists():
        with sample.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        return {"columns": header}
    return None


async def confirm_and_run(args: argparse.Namespace) -> None:
    banner()
    if not args.dry_run:
        load_env()
    else:
        print("Dry run enabled: skipping API key validation and external API calls.")

    jd_path = Path(args.jd).resolve() if args.jd else detect_jd_file(DATA_DIR)
    candidate_path = Path(args.candidates).resolve() if args.candidates else detect_candidate_file(DATA_DIR)
    print(f"JD file: {jd_path}")
    print(f"Candidate file: {candidate_path}")
    print(f"Dry run: {args.dry_run}")
    print(f"Top LLM candidates: {args.top}")
    print("Proceeding in 5 seconds unless interrupted...")
    try:
        await asyncio.wait_for(asyncio.to_thread(input, "Press Enter to continue now: "), timeout=5)
    except Exception:
        pass

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    jd_text = load_jd_text(jd_path)
    jd_extractor = JobDNAExtractor(DATA_DIR)
    job_dna = await jd_extractor.extract(jd_text, use_cache=not args.skip_cache and not args.dry_run, dry_run=args.dry_run)
    timings["jd_extract"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    candidate_df = load_candidates(candidate_path)
    log("info", "Loaded candidates", {"rows": len(candidate_df), "columns": list(candidate_df.columns)})
    profile_builder = CandidateProfileEnricher(DATA_DIR, dry_run=args.dry_run)
    enriched_profiles = await profile_builder.enrich_all(candidate_df)
    timings["profile_enrich"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    scorer = HybridScoringEngine(job_dna, top_n_llm=args.top, dry_run=args.dry_run)
    ranked_results = await scorer.score_all(enriched_profiles)
    timings["scoring"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    generator = RankedOutputGenerator(ROOT / "output")
    paths = generator.generate(ranked_results, output_format_hint=load_output_format_hint(DATA_DIR))
    timings["output"] = time.perf_counter() - t3

    print("\nTiming Summary")
    print("=" * 72)
    for key, value in timings.items():
        print(f"{key:<16}{value:>8.2f}s")
    print(f"\nSubmission file: {paths['ranked']}")
    print(f"Full analysis: {paths['full']}")
    print(f"Score breakdown: {paths['breakdown']}")

    if ranked_results:
        print("\nTop ranked candidate:")
        first = ranked_results[0]
        print(f"{first['candidate_id']} | score={first['final_score']:.4f} | verdict={first.get('fit_verdict')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Talent Intelligence System")
    parser.add_argument("--jd", type=str, default=None, help="Path to JD file")
    parser.add_argument("--candidates", type=str, default=None, help="Path to candidate file")
    parser.add_argument("--skip-cache", action="store_true", help="Force JD extraction even if cached")
    parser.add_argument("--top", type=int, default=40, help="Top N candidates for LLM evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Run without external API calls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(confirm_and_run(args))
    except Exception as exc:
        print(f"Pipeline failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
