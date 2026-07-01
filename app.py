from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from run import DATA_DIR, ROOT, detect_candidate_file, detect_jd_file, load_candidates, load_jd_text, load_output_format_hint
from src.jd_parser import JobDNAExtractor
from src.output import RankedOutputGenerator
from src.profile_builder import CandidateProfileEnricher
from src.scorer import HybridScoringEngine


st.set_page_config(
    page_title="Talent Intelligence System",
    page_icon="\U0001F4CB",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.20), transparent 28%),
                radial-gradient(circle at 85% 10%, rgba(249, 115, 22, 0.16), transparent 22%),
                radial-gradient(circle at bottom left, rgba(59, 130, 246, 0.12), transparent 26%),
                linear-gradient(180deg, #07111c 0%, #0b1324 45%, #111827 100%);
            color: #e5eefb;
        }
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2.2rem;
            max-width: 1540px;
        }
        h1, h2, h3, h4, p, div, span, label {
            color: #e5eefb !important;
        }
        .hero {
            padding: 1.8rem 1.9rem 1.25rem 1.9rem;
            border: 1px solid rgba(255,255,255,0.08);
            background: linear-gradient(135deg, rgba(5, 11, 20, 0.96), rgba(15, 23, 42, 0.74));
            border-radius: 28px;
            box-shadow: 0 28px 80px rgba(0,0,0,0.34);
        }
        .hero h1 {
            font-size: 3.05rem;
            line-height: 0.98;
            letter-spacing: -0.07em;
            margin-bottom: 0.25rem;
        }
        .hero p {
            max-width: 980px;
            font-size: 1.02rem;
            color: rgba(229, 238, 251, 0.80) !important;
        }
        .pill {
            display: inline-block;
            padding: 0.34rem 0.75rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.14);
            background: rgba(255,255,255,0.05);
            font-size: 0.82rem;
            margin-right: 0.45rem;
            margin-bottom: 0.5rem;
        }
        .metric-card {
            padding: 1rem 1rem 0.85rem 1rem;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            backdrop-filter: blur(12px);
        }
        .panel {
            padding: 1rem 1rem 0.8rem 1rem;
            border-radius: 22px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            backdrop-filter: blur(12px);
        }
        .soft {
            color: rgba(229, 238, 251, 0.72) !important;
            font-size: 0.9rem;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_briefs() -> pd.DataFrame:
    path = ROOT / "output" / "candidate_briefs.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return [value]
    return []


def candidate_detail_from_brief(briefs: pd.DataFrame, candidate_id: str) -> dict[str, Any] | None:
    if briefs.empty or "candidate_id" not in briefs.columns:
        return None
    match = briefs[briefs["candidate_id"] == candidate_id]
    if match.empty:
        return None
    row = match.iloc[0].to_dict()
    row["red_flags"] = safe_json_list(row.get("red_flags"))
    row["interview_probes"] = safe_json_list(row.get("interview_probes"))
    return row


def score_bucket(score: float) -> str:
    if score >= 0.62:
        return "Elite"
    if score >= 0.58:
        return "Strong"
    if score >= 0.54:
        return "Promising"
    if score >= 0.50:
        return "Stretch"
    return "Low"


def render_metric_card(title: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="soft">{title}</div>
            <div style="font-size: 1.8rem; font-weight: 700; margin-top: 0.15rem;">{value}</div>
            <div class="soft">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_pipeline(dry_run: bool, top_n: int, jd_path: str, candidate_path: str) -> None:
    job_text = load_jd_text(Path(jd_path))
    if dry_run:
        job_dna = JobDNAExtractor(DATA_DIR)._fallback_extract(job_text)
    else:
        job_dna = None

    candidate_df = load_candidates(Path(candidate_path))
    enricher = CandidateProfileEnricher(DATA_DIR, dry_run=dry_run)
    enriched_records = asyncio.run(enricher.enrich_all(candidate_df))
    scorer = HybridScoringEngine(job_dna or JobDNAExtractor(DATA_DIR)._fallback_extract(job_text), top_n_llm=top_n, dry_run=dry_run)
    ranked_results = asyncio.run(scorer.score_all(enriched_records))
    generator = RankedOutputGenerator(ROOT / "output")
    generator.generate(ranked_results, output_format_hint=load_output_format_hint(DATA_DIR))


def main() -> None:
    apply_styles()

    ranked_df = read_csv(ROOT / "output" / "ranked_candidates.csv")
    breakdown_df = read_csv(ROOT / "output" / "score_breakdown.csv")
    briefs_df = read_briefs()

    st.markdown(
        """
        <div class="hero">
            <div class="pill">Talent Intelligence System</div>
            <div class="pill">Recruiter Decision Studio</div>
            <div class="pill">Hybrid Ranking</div>
            <h1>AI Talent Intelligence Dashboard</h1>
            <p>
                Search candidates, compare finalists, inspect signal breakdowns, and rerun the ranking pipeline from one place.
                The dashboard is built to be useful even in dry-run mode, so you can explore the interface without an API key.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("Controls")
        jd_path = st.text_input("Job description path", value=str(detect_jd_file(DATA_DIR)))
        candidate_path = st.text_input("Candidate file path", value=str(detect_candidate_file(DATA_DIR)))
        top_n = st.slider("Top candidates to LLM-score", min_value=10, max_value=100, value=40, step=5)
        dry_run = st.toggle("Dry run (no API calls)", value=True)
        run_pipeline_button = st.button("Run ranking pipeline", use_container_width=True)
        st.caption("Dry run is the best default for this environment because it avoids external API calls.")

    if run_pipeline_button:
        with st.spinner("Running ranking pipeline..."):
            run_pipeline(dry_run, top_n, jd_path, candidate_path)
        st.success("Pipeline completed. Refreshing the dashboard with the latest outputs.")
        st.rerun()

    if ranked_df.empty:
        st.info("No outputs found yet. Run the pipeline from the sidebar or use `python run.py --dry-run` first.")
        return

    ranked_df = ranked_df.copy()
    ranked_df["score"] = ranked_df["score"].astype(float)
    if not breakdown_df.empty:
        for column in ["semantic_score", "hard_score", "trajectory_bonus", "final_score"]:
            if column in breakdown_df.columns:
                breakdown_df[column] = pd.to_numeric(breakdown_df[column], errors="coerce")

    st.markdown("### Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Ranked candidates", f"{len(ranked_df):,}", "Rows in ranked_candidates.csv")
    with c2:
        render_metric_card("Top score", f"{ranked_df['score'].max():.4f}", "Highest final score")
    with c3:
        verdict_col = "fit_verdict" if "fit_verdict" in ranked_df.columns else None
        verdict_top = ranked_df[verdict_col].mode().iloc[0] if verdict_col and not ranked_df[verdict_col].dropna().empty else "n/a"
        render_metric_card("Dominant verdict", str(verdict_top), "Most common verdict among loaded rows")
    with c4:
        render_metric_card("Score band", score_bucket(float(ranked_df['score'].max())), "Quick high-level bucket")

    tab_rankings, tab_compare, tab_signals, tab_outputs = st.tabs(["Rankings", "Compare", "Signals", "Outputs"])

    with tab_rankings:
        left, right = st.columns([1.5, 0.7])
        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            search = st.text_input("Search candidate, ID, or verdict", value="")
            min_score, max_score = st.slider("Score range", 0.0, 1.0, (0.0, 1.0), 0.01)
            show_top = st.selectbox("Rows to display", [25, 50, 100, 250], index=1)
            filtered = ranked_df[(ranked_df["score"] >= min_score) & (ranked_df["score"] <= max_score)]
            if search:
                needle = search.lower()
                text_cols = [c for c in ["candidate_id", "name", "fit_verdict", "reasoning"] if c in filtered.columns]
                mask = pd.Series(False, index=filtered.index)
                for col in text_cols:
                    mask = mask | filtered[col].astype(str).str.lower().str.contains(needle, na=False)
                filtered = filtered[mask]
            filtered = filtered.head(show_top)
            display_cols = [col for col in ["rank", "candidate_id", "name", "score", "fit_verdict", "reasoning"] if col in filtered.columns]
            st.dataframe(filtered[display_cols], use_container_width=True, height=620)
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            rank_choices = ranked_df["rank"].head(100).tolist()
            selected_rank = st.selectbox("Open candidate detail", rank_choices)
            selected_row = ranked_df[ranked_df["rank"] == selected_rank].iloc[0]
            candidate_id = selected_row["candidate_id"]
            detail = candidate_detail_from_brief(briefs_df, candidate_id)
            st.markdown(f"### {selected_row.get('name', candidate_id)}")
            st.write(f"**Candidate ID:** {candidate_id}")
            st.write(f"**Rank:** {int(selected_row['rank'])}")
            st.write(f"**Final score:** {float(selected_row['score']):.4f}")
            st.write(f"**Verdict:** {selected_row.get('fit_verdict', 'not_evaluated')}")
            if detail:
                st.write(detail.get("reason_for_fit", selected_row.get("reasoning", "")))
                st.write("**Biggest strength**")
                st.write(detail.get("biggest_strength", ""))
                st.write("**Biggest risk**")
                st.write(detail.get("biggest_risk", ""))
                st.write("**Standout signal**")
                st.write(detail.get("standout_signal", ""))
                st.write("**Interview probes**")
                for probe in detail.get("interview_probes", []):
                    st.write(f"- {probe}")
                st.write("**Red flags**")
                if detail.get("red_flags"):
                    for flag in detail.get("red_flags", []):
                        st.write(f"- {flag}")
                else:
                    st.write("None")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_compare:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        options = ranked_df["candidate_id"].head(50).tolist()
        with col_a:
            left_candidate = st.selectbox("Left candidate", options, index=0)
        with col_b:
            right_candidate = st.selectbox("Right candidate", options, index=1 if len(options) > 1 else 0)
        left_row = ranked_df[ranked_df["candidate_id"] == left_candidate].iloc[0]
        right_row = ranked_df[ranked_df["candidate_id"] == right_candidate].iloc[0]
        compare_df = pd.DataFrame(
            [
                {"candidate": left_row.get("name", left_candidate), "rank": left_row["rank"], "score": left_row["score"], "verdict": left_row.get("fit_verdict", "n/a")},
                {"candidate": right_row.get("name", right_candidate), "rank": right_row["rank"], "score": right_row["score"], "verdict": right_row.get("fit_verdict", "n/a")},
            ]
        )
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
        if not breakdown_df.empty:
            merged = breakdown_df[breakdown_df["candidate_id"].isin([left_candidate, right_candidate])].copy()
            merged = merged[[c for c in ["candidate_id", "semantic_score", "hard_score", "llm_score", "trajectory_bonus", "final_score"] if c in merged.columns]]
            st.dataframe(merged, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_signals:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Signal distribution")
        if not breakdown_df.empty:
            chart_df = breakdown_df[[c for c in ["rank", "semantic_score", "hard_score", "trajectory_bonus", "final_score"] if c in breakdown_df.columns]].copy()
            st.dataframe(chart_df.head(50), use_container_width=True, hide_index=True)
        else:
            st.info("Signal breakdown is not available yet.")
        st.subheader("Verdict mix")
        if "fit_verdict" in ranked_df.columns:
            verdict_counts = ranked_df["fit_verdict"].fillna("unknown").value_counts().reset_index()
            verdict_counts.columns = ["verdict", "count"]
            st.dataframe(verdict_counts, use_container_width=True, hide_index=True)
        st.markdown("### Fast filters")
        one, two, three, four = st.columns(4)
        with one:
            st.write("Top 5 names")
            for item in ranked_df.head(5).itertuples():
                display_name = getattr(item, "name", item.candidate_id)
                st.write(f"{item.rank}. {display_name}")
        with two:
            st.write("Score quartiles")
            st.write(ranked_df["score"].quantile([0.25, 0.5, 0.75]).to_frame("score"))
        with three:
            st.write("Candidates above 0.60")
            st.write(int((ranked_df["score"] >= 0.60).sum()))
        with four:
            st.write("Candidates below 0.50")
            st.write(int((ranked_df["score"] < 0.50).sum()))
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_outputs:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.write("These are the files the pipeline writes to `output/`.")
        output_dir = ROOT / "output"
        files = [
            output_dir / "ranked_candidates.csv",
            output_dir / "score_breakdown.csv",
            output_dir / "candidate_briefs.csv",
            output_dir / "full_analysis.json",
        ]
        file_rows = []
        for file in files:
            if file.exists():
                file_rows.append({"file": file.name, "size_kb": round(file.stat().st_size / 1024, 1)})
        st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)
        st.markdown("### Run commands")
        st.code("python run.py --dry-run\nstreamlit run app.py", language="bash")
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()