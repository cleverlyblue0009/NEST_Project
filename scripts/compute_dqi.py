"""
STEP 3: DATA QUALITY INDEX (DQI) COMPUTATION + ML ANOMALY DETECTION
===================================================================

Changes from original:
  1. Import anomaly detection module (NEW)
  2. Call detect_site_anomalies() after computing rule-based DQI
  3. Amplify risk for anomalous sites
  4. Save anomaly scores to output
  5. Use amplified DQI for downstream risk categorization

All rule-based DQI logic remains unchanged.
ML only augments risk for anomalous sites.
"""

import os
import pandas as pd
import numpy as np

# NEW: Import anomaly detection module
try:
    from detect_anomalies import detect_site_anomalies, amplify_risk_for_anomalies
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    print("⚠ WARNING: detect_anomalies module not found. Running without ML enhancement.")
    print("  To enable anomaly detection, ensure detect_anomalies.py is in scripts/ folder")
    ANOMALY_DETECTION_AVAILABLE = False


# ============================================================================
# CLINICAL THRESHOLDS (from earlier fixes)
# ============================================================================

CLINICAL_THRESHOLDS = {
    'missing_pages_pct': 5.0,
    'missing_visits_pct': 10.0,
    'unresolved_edrr_pct': 8.0,
    'uncoded_terms_pct': 7.0,
    'pending_sae_pct': 5.0,
}


def normalize_signal_vs_threshold(signal_pct, threshold_pct):
    """
    Normalize signal relative to clinical threshold (not global max).
    
    Args:
        signal_pct (float): Signal percentage
        threshold_pct (float): Clinical acceptance threshold
    
    Returns:
        float: Normalized penalty (0-100)
    """
    if signal_pct <= threshold_pct:
        return 0.0
    else:
        excess = signal_pct - threshold_pct
        normalized = (excess / threshold_pct) * 100
        return min(normalized, 100.0)


def compute_dqi_study_level(study_signals):
    """
    Compute study-level DQI using threshold-based normalization.
    
    Args:
        study_signals (pd.DataFrame): Study-level signal percentages
    
    Returns:
        pd.DataFrame: DQI scores with penalty breakdown
    """
    
    dqi_df = study_signals.copy()
    
    W_pages, W_visits, W_edrr, W_codes, W_sae = 10, 15, 15, 12, 25
    total_weight = W_pages + W_visits + W_edrr + W_codes + W_sae
    
    # Normalize each signal relative to clinical threshold
    pct_pages = study_signals['missing_pages_pct'].apply(
        lambda x: normalize_signal_vs_threshold(x, CLINICAL_THRESHOLDS['missing_pages_pct'])
    )
    pct_visits = study_signals['missing_visits_pct'].apply(
        lambda x: normalize_signal_vs_threshold(x, CLINICAL_THRESHOLDS['missing_visits_pct'])
    )
    pct_edrr = study_signals['unresolved_edrr_pct'].apply(
        lambda x: normalize_signal_vs_threshold(x, CLINICAL_THRESHOLDS['unresolved_edrr_pct'])
    )
    pct_codes = study_signals['uncoded_terms_pct'].apply(
        lambda x: normalize_signal_vs_threshold(x, CLINICAL_THRESHOLDS['uncoded_terms_pct'])
    )
    pct_sae = study_signals['pending_sae_pct'].apply(
        lambda x: normalize_signal_vs_threshold(x, CLINICAL_THRESHOLDS['pending_sae_pct'])
    )
    
    # Compute weighted penalty
    weighted_penalty = (
        (pct_pages * W_pages) +
        (pct_visits * W_visits) +
        (pct_edrr * W_edrr) +
        (pct_codes * W_codes) +
        (pct_sae * W_sae)
    ) / total_weight
    
    # DQI = 100 - penalty
    dqi_df['dqi_score'] = 100 - weighted_penalty
    
    # Store penalty breakdown for transparency
    dqi_df['penalty_pages'] = pct_pages * (W_pages / total_weight)
    dqi_df['penalty_visits'] = pct_visits * (W_visits / total_weight)
    dqi_df['penalty_edrr'] = pct_edrr * (W_edrr / total_weight)
    dqi_df['penalty_codes'] = pct_codes * (W_codes / total_weight)
    dqi_df['penalty_sae'] = pct_sae * (W_sae / total_weight)
    
    # Round for readability
    dqi_df['dqi_score'] = dqi_df['dqi_score'].round(2)
    for col in ['penalty_pages', 'penalty_visits', 'penalty_edrr', 'penalty_codes', 'penalty_sae']:
        dqi_df[col] = dqi_df[col].round(2)
    
    return dqi_df


def compute_dqi_site_level(site_signals):
    """
    Compute site-level DQI using threshold-based normalization.
    
    Args:
        site_signals (pd.DataFrame): Site-level signal percentages
    
    Returns:
        pd.DataFrame: Site DQI scores with penalty breakdown
    """
    
    dqi_df = site_signals.copy()
    
    W_pages, W_visits, W_edrr, W_codes, W_sae = 10, 15, 15, 12, 25
    total_weight = W_pages + W_visits + W_edrr + W_codes + W_sae
    
    # Normalize relative to clinical thresholds
    pct_pages = site_signals['missing_pages_pct'].apply(
        lambda x: normalize_signal_vs_threshold(x, CLINICAL_THRESHOLDS['missing_pages_pct'])
    )
    pct_visits = site_signals['missing_visits_pct'].apply(
        lambda x: normalize_signal_vs_threshold(x, CLINICAL_THRESHOLDS['missing_visits_pct'])
    )
    pct_edrr = site_signals['unresolved_edrr_pct'].apply(
        lambda x: normalize_signal_vs_threshold(x, CLINICAL_THRESHOLDS['unresolved_edrr_pct'])
    )
    pct_codes = site_signals['uncoded_terms_pct'].apply(
        lambda x: normalize_signal_vs_threshold(x, CLINICAL_THRESHOLDS['uncoded_terms_pct'])
    )
    pct_sae = site_signals['pending_sae_pct'].apply(
        lambda x: normalize_signal_vs_threshold(x, CLINICAL_THRESHOLDS['pending_sae_pct'])
    )
    
    weighted_penalty = (
        (pct_pages * W_pages) +
        (pct_visits * W_visits) +
        (pct_edrr * W_edrr) +
        (pct_codes * W_codes) +
        (pct_sae * W_sae)
    ) / total_weight
    
    dqi_df['dqi_score'] = 100 - weighted_penalty
    
    dqi_df['penalty_pages'] = pct_pages * (W_pages / total_weight)
    dqi_df['penalty_visits'] = pct_visits * (W_visits / total_weight)
    dqi_df['penalty_edrr'] = pct_edrr * (W_edrr / total_weight)
    dqi_df['penalty_codes'] = pct_codes * (W_codes / total_weight)
    dqi_df['penalty_sae'] = pct_sae * (W_sae / total_weight)
    
    dqi_df['dqi_score'] = dqi_df['dqi_score'].round(2)
    for col in ['penalty_pages', 'penalty_visits', 'penalty_edrr', 'penalty_codes', 'penalty_sae']:
        dqi_df[col] = dqi_df[col].round(2)
    
    return dqi_df


def print_dqi_summary(study_dqi):
    """Print summary of DQI methodology and results."""
    
    print("\n" + "=" * 80)
    print("DATA QUALITY INDEX (DQI) SUMMARY")
    print("=" * 80)
    print("\nDQI Scoring Methodology (Rule-Based):")
    print("  DQI = 100 - [weighted penalty from signals vs clinical thresholds]")
    print("  Higher DQI = Better data quality")
    print("\nWeights (by clinical impact):")
    print("  CRF Pages (W=10):     Completeness foundation")
    print("  Missing Visits (W=15): Safety data gaps; blocks submission")
    print("  EDRR Queries (W=15):   Delays database lock")
    print("  Uncoded Terms (W=12):  Safety/efficacy ambiguity")
    print("  SAE Reviews (W=25):    Highest regulatory/safety priority")
    print("\nInterpretation:")
    print("  DQI >= 80: Excellent (all signals at/below threshold)")
    print("  DQI 60-80: Good (some signals moderately above threshold)")
    print("  DQI 40-60: Fair (multiple signals significantly above threshold)")
    print("  DQI < 40:  Poor (most signals well above threshold)")
    print("\n" + "=" * 80)
    
    print("\nSTUDY-LEVEL DQI SCORES:")
    display_cols = ['study_id', 'dqi_score', 'missing_pages_pct', 'missing_visits_pct', 'unresolved_edrr_pct']
    print(study_dqi[display_cols].to_string(index=False))
    
    avg_dqi = study_dqi['dqi_score'].mean()
    print(f"\nAverage DQI: {avg_dqi:.2f}")


def save_dqi_scores(study_dqi, site_dqi, output_path="outputs/dqi_scores.csv"):
    """Save DQI scores to CSV files."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    study_dqi.to_csv(output_path, index=False)
    print(f"✓ Study-level DQI saved to: {output_path}")
    
    site_path = output_path.replace('dqi_scores.csv', 'dqi_scores_site_level.csv')
    site_dqi.to_csv(site_path, index=False)
    print(f"✓ Site-level DQI saved to: {site_path}")
    
    return study_dqi, site_dqi


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STEP 3: DATA QUALITY INDEX (DQI) COMPUTATION")
    print("=" * 80)
    
    # Load signals
    study_signals_path = "outputs/signals_study_level.csv"
    site_signals_path = "outputs/signals_site_level.csv"
    
    if not os.path.exists(study_signals_path) or not os.path.exists(site_signals_path):
        print(f"\n❌ Signal files not found. Run extract_signals.py first.")
    else:
        study_signals = pd.read_csv(study_signals_path)
        site_signals = pd.read_csv(site_signals_path)
        
        print(f"\nLoaded signals for {len(study_signals)} studies and {len(site_signals)} sites")
        
        # Compute rule-based DQI
        print("\nComputing rule-based DQI scores...")
        study_dqi = compute_dqi_study_level(study_signals)
        site_dqi = compute_dqi_site_level(site_signals)
        
        # =====================================================================
        # NEW: ANOMALY DETECTION & RISK AMPLIFICATION
        # =====================================================================
        if ANOMALY_DETECTION_AVAILABLE:
            print("\n" + "=" * 80)
            print("ANOMALY DETECTION & RISK AMPLIFICATION (ML Enhancement)")
            print("=" * 80)
            
            # Detect anomalous sites
            anomalies_site = detect_site_anomalies(site_dqi)
            
            # Amplify risk for anomalous sites
            site_dqi = amplify_risk_for_anomalies(site_dqi, anomalies_site)
            
            # Save anomaly scores
            anomalies_site.to_csv('outputs/anomalies_site_level.csv', index=False)
            print(f"✓ Anomaly scores saved to: outputs/anomalies_site_level.csv")
            
            # Also merge anomaly info into study level
            anomalies_summary = anomalies_site.groupby('study_id').agg({
                'anomaly_score': 'mean',
                'is_anomalous': 'sum'
            }).reset_index()
            anomalies_summary.columns = ['study_id', 'avg_anomaly_score', 'num_anomalous_sites']
            
            study_dqi = study_dqi.merge(anomalies_summary, on='study_id', how='left')
            study_dqi['avg_anomaly_score'] = study_dqi['avg_anomaly_score'].fillna(0)
            study_dqi['num_anomalous_sites'] = study_dqi['num_anomalous_sites'].fillna(0)
        else:
            print("\n⚠ Skipping anomaly detection (module not available)")
            # Add placeholder columns for consistency
            site_dqi['dqi_score_amplified'] = site_dqi['dqi_score']
        
        # =====================================================================
        # Save DQI scores
        # =====================================================================
        save_dqi_scores(study_dqi, site_dqi)
        
        # Print summary
        print_dqi_summary(study_dqi)
        print("\n✓ DQI computation complete.\n")