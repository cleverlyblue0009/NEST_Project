# src/ai_insights_generator.py
# OpenAI API Integration for Generative Insights

import json
import os
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class AIInsightsGenerator:
    """
    Generates AI-powered insights using OpenAI for clinical trial data.
    Stable, quota-safe, and production-grade.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Set it in .env or environment variables."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"

    # ------------------------------------------------------------
    # EXECUTIVE SUMMARY
    # ------------------------------------------------------------

    def generate_executive_summary(
        self,
        study_id: str,
        dqi_score: float,
        risk_level: str,
        metrics: Dict,
        primary_driver: str,
        site_count: int,
        enrollment: int,
    ) -> str:

        prompt = f"""
You are a Clinical Data Quality Expert analyzing pharmaceutical trial data.

STUDY DATA
- Study ID: {study_id}
- Data Quality Index (DQI): {dqi_score}/100
- Risk Level: {risk_level}
- Sites: {site_count}
- Enrolled Subjects: {enrollment}
- Primary Quality Driver: {primary_driver}

QUALITY METRICS
- Missing Visits: {metrics['missing_visits_pct']}%
- Missing Pages: {metrics['missing_pages_pct']}%
- Open Queries: {metrics['open_queries_pct']}%
- Unverified Forms: {metrics['unverified_forms_pct']}%
- Uncoded Terms: {metrics['uncoded_terms_pct']}%

TASK
Write a 2–3 sentence executive summary that:
1. States overall data quality risk
2. Identifies the main blocker
3. Recommends immediate action

Use professional pharmaceutical language.
Be direct and actionable.
Do NOT include headings or bullet points.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a senior clinical data quality analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=180,
        )

        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------
    # SITE PERFORMANCE REPORT
    # ------------------------------------------------------------

    def generate_site_performance_report(
        self,
        study_id: str,
        site_id: str,
        cra_name: str,
        enrollment: int,
        open_queries: int,
        missing_pages: int,
        query_rate: float,
        days_since_last_visit: int,
    ) -> str:

        prompt = f"""
You are a CRA performance analyst.

SITE DATA
- Study: {study_id}
- Site ID: {site_id}
- CRA: {cra_name}
- Enrollment: {enrollment}
- Open Queries: {open_queries}
- Missing Pages: {missing_pages}
- Query Rate: {query_rate}% (benchmark: 6%)
- Days Since Last Monitor Visit: {days_since_last_visit}

TASK
Generate a concise site performance report with:
- Performance summary
- Top 2 issues with actions
- Monitoring recommendation
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a clinical operations performance reviewer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.25,
            max_tokens=200,
        )

        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------
    # RISK ALERT
    # ------------------------------------------------------------

    def generate_risk_alert(
        self,
        study_id: str,
        dqi_previous: float,
        dqi_current: float,
        trend: str,
        critical_issues: List[str],
        days_to_interim_analysis: int,
    ) -> str:

        issues_text = "\n".join(f"- {i}" for i in critical_issues)

        prompt = f"""
You are a clinical trial risk manager.

STUDY
- Study: {study_id}
- Previous DQI: {dqi_previous}
- Current DQI: {dqi_current}
- Trend: {trend.upper()}
- Days to Interim Analysis: {days_to_interim_analysis}

CRITICAL ISSUES
{issues_text}

TASK
If risk is high and interim < 30 days, write an URGENT alert.
Otherwise, write a STANDARD alert.

2–3 sentences.
Start with 🚨 if urgent, ⚠️ if standard.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You specialize in clinical trial risk escalation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=150,
        )

        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------
    # BATCH PROCESSING
    # ------------------------------------------------------------

    def generate_batch_insights(self, ranked_studies: Dict) -> Dict[str, Dict]:
        insights = {}

        for i, (study_id, data) in enumerate(list(ranked_studies.items())[:10], 1):
            print(f"Generating insights for {study_id} ({i}/10)")

            study = data["study_level"]

            exec_summary = self.generate_executive_summary(
                study_id=study_id,
                dqi_score=study["dqi_score"],
                risk_level=study["risk_level"],
                metrics=study["metrics"],
                primary_driver=study["primary_driver"],
                site_count=study["site_count"],
                enrollment=study["enrollment_count"],
            )

            prev_dqi = study.get("previous_dqi", study["dqi_score"])

            risk_alert = self.generate_risk_alert(
                study_id=study_id,
                dqi_previous=prev_dqi,
                dqi_current=study["dqi_score"],
                trend="declining" if study["dqi_score"] < 75 else "stable",
                critical_issues=[study["primary_driver"]],
                days_to_interim_analysis=14 if study["risk_level"] == "HIGH RISK" else 30,
            )

            insights[study_id] = {
                "dqi_score": study["dqi_score"],
                "risk_level": study["risk_level"],
                "primary_driver": study["primary_driver"],
                "executive_summary": exec_summary,
                "risk_alert": risk_alert,
                "metrics": study["metrics"],
            }

        return insights

    # ------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------

    def export_insights(self, insights: Dict, output_file: str):
        with open(output_file, "w") as f:
            json.dump(insights, f, indent=2)
        print(f"✓ Insights exported to {output_file}")


class BatchInsightsProcessor:
    def __init__(self):
        self.generator = AIInsightsGenerator()
