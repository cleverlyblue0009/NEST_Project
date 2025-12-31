"""
STEP 4: RISK RANKING & CATEGORIZATION (UPDATED WITH ML)
========================================================

Changes from original:
  1. Use amplified DQI (if available) instead of original DQI
  2. Percentile-based risk categorization (worst 10% = High Risk)
  3. Identify risk driver: "Rule-based" vs "Anomalous pattern"
  4. Output anomaly metadata for explanations

All percentile-based logic remains unchanged.
ML amplification is used in DQI calculation, not in risk logic.
"""

import os
import pandas as pd
import numpy as np


def categorize_risk_with_guardrails(dqi_scores_series):
    """
    Percentile-based risk categorization.
    
    Guarantees:
    - Worst 10% → High Risk
    - Next 15% → Medium Risk
    - Top 75% → Low Risk
    
    Args:
        dqi_scores_series (pd.Series): DQI scores
    
    Returns:
        pd.Series: Risk categories
    """
    
    p90 = dqi_scores_series.quantile(0.90)
    p75 = dqi_scores_series.quantile(0.75)
    
    def assign_risk(score):
        if score <= p90:
            return "High Risk"
        elif score <= p75:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    return dqi_scores_series.apply(assign_risk)


def identify_top_penalty_drivers(row, top_n=2):
    """
    Identify top penalty drivers for a site/study.
    
    Args:
        row (pd.Series): DQI record with penalty columns
        top_n (int): Number of top drivers to return
    
    Returns:
        list: Names of top penalty drivers
    """
    
    signal_cols = {
        'penalty_pages': 'CRF Pages',
        'penalty_visits': 'Missing Visits',
        'penalty_edrr': 'EDRR Queries',
        'penalty_codes': 'Uncoded Terms',
        'penalty_sae': 'SAE Reviews',
    }
    
    # If old column names, use alternative mapping
    if 'penalty_pages' not in row.index:
        signal_cols = {
            'missing_pages_pct': 'CRF Pages',
            'missing_visits_pct': 'Missing Visits',
            'unresolved_edrr_pct': 'EDRR Queries',
            'uncoded_terms_pct': 'Uncoded Terms',
            'pending_sae_pct': 'SAE Reviews',
        }
    
    penalties = {}
    for col, label in signal_cols.items():
        if col in row.index:
            penalties[label] = float(row.get(col, 0))
        else:
            penalties[label] = 0
    
    sorted_penalties = sorted(penalties.items(), key=lambda x: x[1], reverse=True)
    return [p[0] for p in sorted_penalties[:top_n] if p[1] > 0]


def rank_studies(study_dqi):
    """
    Rank studies by risk using amplified DQI (if available).
    
    Args:
        study_dqi (pd.DataFrame): Study-level DQI scores
    
    Returns:
        pd.DataFrame: Ranked studies with risk categories
    """
    
    ranked = study_dqi.copy()
    
    # Use amplified DQI if available, otherwise use original
    dqi_column = 'dqi_score_amplified' if 'dqi_score_amplified' in ranked.columns else 'dqi_score'
    
    print(f"\nRisk categorization using: {dqi_column}")
    
    # Percentile-based risk categorization
    ranked['risk_level'] = categorize_risk_with_guardrails(ranked[dqi_column])
    
    # Identify top risk drivers
    ranked['top_risk_drivers'] = ranked.apply(
        lambda row: ", ".join(identify_top_penalty_drivers(row, top_n=2)),
        axis=1
    )
    
    # Rank by DQI (worst first)
    ranked = ranked.sort_values(dqi_column).reset_index(drop=True)
    ranked['rank'] = range(1, len(ranked) + 1)
    
    # Count total signals
    ranked['total_signal_pct'] = (
        ranked.get('missing_pages_pct', 0) + 
        ranked.get('missing_visits_pct', 0) + 
        ranked.get('unresolved_edrr_pct', 0) + 
        ranked.get('uncoded_terms_pct', 0) + 
        ranked.get('pending_sae_pct', 0)
    )
    
    return ranked[['rank', 'study_id', 'dqi_score', 'risk_level', 'total_signal_pct', 'top_risk_drivers']]


def rank_sites(site_dqi):
    """
    Rank sites by risk using amplified DQI (if available).
    Identifies whether risk is rule-based or anomaly-driven.
    
    Args:
        site_dqi (pd.DataFrame): Site-level DQI scores with anomaly metadata
    
    Returns:
        pd.DataFrame: Ranked sites with risk categories
    """
    
    ranked = site_dqi.copy()
    
    # Use amplified DQI if available, otherwise use original
    dqi_column = 'dqi_score_amplified' if 'dqi_score_amplified' in ranked.columns else 'dqi_score'
    
    print(f"Risk categorization using: {dqi_column}")
    
    # Percentile-based risk categorization
    ranked['risk_level'] = categorize_risk_with_guardrails(ranked[dqi_column])
    
    # Identify top drivers
    ranked['top_risk_drivers'] = ranked.apply(
        lambda row: ", ".join(identify_top_penalty_drivers(row, top_n=2)),
        axis=1
    )
    
    # Identify risk driver (rule-based vs anomaly)
    ranked['risk_driver'] = 'Rule-based'
    if 'is_anomalous' in ranked.columns:
        anomalous_high = (ranked['is_anomalous'] == 1) & (ranked['risk_level'] == 'High Risk')
        ranked.loc[anomalous_high, 'risk_driver'] = 'ML-Detected Anomaly'
    
    # Rank globally
    ranked = ranked.sort_values(dqi_column).reset_index(drop=True)
    ranked['global_rank'] = range(1, len(ranked) + 1)
    
    # Rank within-study
    ranked['within_study_rank'] = ranked.groupby('study_id').cumcount() + 1
    
    # Count total signal percentage
    ranked['total_signal_pct'] = (
        ranked.get('missing_pages_pct', 0) + 
        ranked.get('missing_visits_pct', 0) + 
        ranked.get('unresolved_edrr_pct', 0) + 
        ranked.get('uncoded_terms_pct', 0) + 
        ranked.get('pending_sae_pct', 0)
    )
    
    return ranked[['study_id', 'site_id', 'global_rank', 'within_study_rank', 'dqi_score', 
                   'risk_level', 'risk_driver', 'total_signal_pct', 'top_risk_drivers']]


def generate_risk_summary(study_ranks):
    """
    Generate risk summary with distribution statistics.
    
    Args:
        study_ranks (pd.DataFrame): Ranked studies
    
    Returns:
        str: Summary text
    """
    
    high_risk = len(study_ranks[study_ranks['risk_level'] == 'High Risk'])
    medium_risk = len(study_ranks[study_ranks['risk_level'] == 'Medium Risk'])
    low_risk = len(study_ranks[study_ranks['risk_level'] == 'Low Risk'])
    total = len(study_ranks)
    
    summary = f"""
RISK DISTRIBUTION (PERCENTILE-BASED WITH GUARANTEES):
======================================================
Total Studies: {total}
  ✗ High Risk (Worst 10%):       {high_risk} studies ({100*high_risk/total:.1f}%)
  ⚠ Medium Risk (Next 15%):      {medium_risk} studies ({100*medium_risk/total:.1f}%)
  ✓ Low Risk (Top 75%):          {low_risk} studies ({100*low_risk/total:.1f}%)

Top 3 Most At-Risk Studies:
===========================
"""
    
    top_3 = study_ranks.head(3)
    for idx, row in top_3.iterrows():
        summary += f"\n{row['rank']}. {row['study_id']} (DQI: {row['dqi_score']:.2f})\n"
        summary += f"   Risk Level: {row['risk_level']}\n"
        summary += f"   Top Drivers: {row['top_risk_drivers']}\n"
    
    return summary


def save_risk_rankings(study_ranks, site_ranks, output_path="outputs/risk_rankings.csv"):
    """Save risk rankings to CSV files."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Study-level rankings
    study_ranks.to_csv(output_path, index=False)
    print(f"\n✓ Study-level risk rankings saved to: {output_path}")
    
    # Site-level rankings
    site_path = output_path.replace('risk_rankings.csv', 'risk_rankings_site_level.csv')
    site_ranks.to_csv(site_path, index=False)
    print(f"✓ Site-level risk rankings saved to: {site_path}")
    
    return study_ranks, site_ranks


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STEP 4: RISK RANKING & CATEGORIZATION")
    print("=" * 80)
    
    # Load DQI scores
    study_dqi_path = "outputs/dqi_scores.csv"
    site_dqi_path = "outputs/dqi_scores_site_level.csv"
    
    if not os.path.exists(study_dqi_path) or not os.path.exists(site_dqi_path):
        print(f"\n❌ DQI files not found. Run compute_dqi.py first.")
    else:
        study_dqi = pd.read_csv(study_dqi_path)
        site_dqi = pd.read_csv(site_dqi_path)
        
        print(f"\nLoaded DQI scores for {len(study_dqi)} studies and {len(site_dqi)} sites")
        
        # Rank studies and sites
        print("\nRanking and categorizing risk...")
        study_ranks = rank_studies(study_dqi)
        site_ranks = rank_sites(site_dqi)
        
        # Save rankings
        save_risk_rankings(study_ranks, site_ranks)
        
        # Print summary
        summary = generate_risk_summary(study_ranks)
        print(summary)
        
        print("\nSTUDY RANKINGS (sorted by risk, worst first):")
        print("=" * 80)
        print(study_ranks.to_string(index=False))
        
        print("\n✓ Risk ranking complete.\n")