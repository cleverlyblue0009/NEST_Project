# src/metrics_calculator.py
# FINAL FIXED VERSION – REAL-WORLD ROBUST DQI

import pandas as pd
from pathlib import Path
from typing import Dict
from datetime import datetime
import json

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def find_file(study_dir: Path, keywords):
    for f in study_dir.iterdir():
        name = f.name.lower()
        if all(k in name for k in keywords):
            return f
    return None


def read_file(f: Path):
    if f.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(f)
    return pd.read_csv(f)


def find_column(df: pd.DataFrame, keywords):
    for col in df.columns:
        name = col.lower()
        if all(k in name for k in keywords):
            return col
    return None


def safe_pct(numer, denom, penalty=25):
    if denom and denom > 0:
        return round((numer / denom) * 100, 2)
    return penalty  # penalize missing denominator


# --------------------------------------------------
# DQI Calculator
# --------------------------------------------------

class DQICalculator:

    WEIGHTS = {
        "visits": 0.25,
        "pages": 0.25,
        "queries": 0.20,
        "forms": 0.15,
        "coding": 0.15
    }

    def calculate(self, m: Dict[str, float]) -> float:
        score = (
            (100 - m["missing_visits_pct"]) * self.WEIGHTS["visits"] +
            (100 - m["missing_pages_pct"]) * self.WEIGHTS["pages"] +
            (100 - m["open_queries_pct"]) * self.WEIGHTS["queries"] +
            (100 - m["unverified_forms_pct"]) * self.WEIGHTS["forms"] +
            (100 - m["uncoded_terms_pct"]) * self.WEIGHTS["coding"]
        )
        return round(max(0, min(100, score)), 2)

    def classify(self, dqi):
        if dqi < 70:
            return "HIGH RISK"
        elif dqi < 80:
            return "MEDIUM RISK"
        return "LOW RISK"


# --------------------------------------------------
# Batch Processor
# --------------------------------------------------

class BatchDQIProcessor:

    def __init__(self):
        self.calc = DQICalculator()

    def process_all_studies(self, raw_dir: str) -> Dict:
        raw_dir = Path(raw_dir)
        results = {}

        for study_dir in raw_dir.iterdir():
            if not study_dir.is_dir():
                continue

            print(f"Processing {study_dir.name}")

            # ---------------- VISITS ----------------
            visit_file = find_file(study_dir, ["visit"])
            missing_visits = expected_visits = 0

            if visit_file:
                df = read_file(visit_file)
                expected_visits = len(df)

                status_col = find_column(df, ["status"]) or find_column(df, ["completion"])
                if status_col:
                    missing_visits = df[df[status_col].astype(str)
                        .str.contains("miss|not|incomplete", case=False, na=False)].shape[0]

            # ---------------- PAGES ----------------
            pages_file = find_file(study_dir, ["missing", "page"])
            missing_pages = total_pages = 0

            if pages_file:
                df = read_file(pages_file)
                missing_pages = len(df)
                total_pages = max(len(df) * 5, 1)  # conservative

            # ---------------- QUERIES ----------------
            query_file = find_file(study_dir, ["query"])
            open_queries = total_queries = 0

            if query_file:
                df = read_file(query_file)
                total_queries = len(df)

                status_col = find_column(df, ["status"])
                if status_col:
                    open_queries = df[df[status_col].astype(str)
                        .str.contains("open|pending", case=False, na=False)].shape[0]

            # ---------------- FORMS ----------------
            form_file = find_file(study_dir, ["inactivated", "form"])
            unverified_forms = total_forms = 0

            if form_file:
                df = read_file(form_file)
                total_forms = len(df)
                unverified_forms = total_forms

            # ---------------- CODING ----------------
            uncoded = total_terms = 0
            for f in [
                find_file(study_dir, ["meddra"]),
                find_file(study_dir, ["whodrug"])
            ]:
                if f:
                    df = read_file(f)
                    total_terms += len(df)
                    status_col = find_column(df, ["status"])
                    if status_col:
                        uncoded += df[df[status_col].astype(str)
                            .str.contains("uncoded|pending", case=False, na=False)].shape[0]

            # ---------------- METRICS ----------------
            metrics = {
                "missing_visits_pct": safe_pct(missing_visits, expected_visits),
                "missing_pages_pct": safe_pct(missing_pages, total_pages),
                "open_queries_pct": safe_pct(open_queries, total_queries),
                "unverified_forms_pct": safe_pct(unverified_forms, total_forms),
                "uncoded_terms_pct": safe_pct(uncoded, total_terms)
            }

            dqi = self.calc.calculate(metrics)

            results[study_dir.name] = {
                "study_level": {
                    "dqi_score": dqi,
                    "risk_level": self.calc.classify(dqi),
                    "metrics": metrics,
                    "primary_driver": max(metrics, key=metrics.get),
                    "site_count": expected_visits,
                    "enrollment_count": expected_visits,
                    "timestamp": datetime.now().isoformat()
                }
            }

        return results

    def export_results(self, output_file: str, results: dict):
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

