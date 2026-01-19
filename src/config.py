# src/config.py
# Central configuration for NEST 2.0

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
OUTPUT_DIR = DATA_DIR / 'outputs'
VISUALIZATIONS_DIR = OUTPUT_DIR / 'visualizations'

# Create directories if they don't exist
for directory in [PROCESSED_DATA_DIR, OUTPUT_DIR, VISUALIZATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# API CONFIGURATION
# ============================================================================

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
    except ImportError:
        print("Warning: python-dotenv not installed. Set ANTHROPIC_API_KEY manually.")

CLAUDE_MODEL = "claude-opus-4.1"
CLAUDE_MAX_TOKENS = 500

# ============================================================================
# DQI CONFIGURATION
# ============================================================================

# DQI Component Weights (must sum to 1.0)
DQI_WEIGHTS = {
    'visits': 0.25,           # Visit Completeness
    'pages': 0.25,            # Form Data Quality
    'queries': 0.20,          # Query Resolution
    'verification': 0.15,     # Form Verification (SDV)
    'safety': 0.15            # Safety/Coding Completeness
}

# Risk Thresholds (DQI Score boundaries)
RISK_THRESHOLDS = {
    'critical': (0, 60),
    'high': (60, 70),
    'medium': (70, 80),
    'low': (80, 100)
}

# Risk Level Labels
RISK_LABELS = {
    'critical': '🔴🔴 CRITICAL RISK',
    'high': '🔴 HIGH RISK',
    'medium': '🟡 MEDIUM RISK',
    'low': '🟢 LOW RISK'
}

# ============================================================================
# DATA PROCESSING CONFIGURATION
# ============================================================================

# Expected CSV columns for EDC metrics
EXPECTED_EDC_COLUMNS = [
    'Project Name',
    'Region',
    'Country',
    'Site ID',
    'Subject ID',
    'Missing Visits',
    '# Expected Visits (Rave EDC : BO4)',
    'Missing Page',
    '# Pages Entered',
    '# Coded terms',
    '# Uncoded Terms',
    '# Open issues in LNR',
    '# Open Issues reported for 3rd party reconciliation in EDRR',
    'Inactivated forms and folders',
    '# eSAE dashboard review for DM',
    '# eSAE dashboard review for safety',
    'Visit status',
    'Page status (Source: (Rave EDC : BO4))',
    'Queries status (Source:(Rave EDC : BO4))',
    'Page Action Status (Source: (Rave EDC : BO4))',
    'Protocol Deviations (Source:(Rave EDC : BO4))',
    'PI Signatures (Source: (Rave EDC : BO4))',
    '# Pages with Non-Conformant data',
    '# Total CRFs with queries & Non-Conformant data',
    '# Total CRFs without queries & Non-Conformant data',
    '% Clean Entered CRF',
    '# DM Queries',
    '# Clinical Queries',
    '# Medical Queries',
    '# Site Queries',
    '# Field Monitor Queries',
    '# Coding Queries',
    '# Safety Queries',
    '#Total Queries',
    '# CRFs Require Verification (SDV)',
    '# Forms Verified',
    '# CRFs Frozen',
    '# CRFs Locked',
    '# CRFs Signed',
    'CRFs overdue for signs within 45 days of Data entry',
    'CRFs overdue for signs between 45 to 90 days of Data entry',
    'CRFs overdue for signs beyond 90 days of Data entry'
]

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Plot DPI (resolution for saved images)
PLOT_DPI = 300

# Color scheme
COLORS = {
    'critical': '#8B0000',    # Dark red
    'high': '#EF553B',         # Red
    'medium': '#FFA15A',       # Orange
    'low': '#00CC96',          # Green
    'primary': '#636EFA',      # Blue
    'secondary': '#AB63FA'     # Purple
}

# Plot style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# ============================================================================
# REPORTING CONFIGURATION
# ============================================================================

# Number of top-risk studies to analyze in detail
TOP_STUDIES_COUNT = 10

# Number of sites per study to show in reports
TOP_SITES_PER_STUDY = 5

# Number of patients per site to show in reports
TOP_PATIENTS_PER_SITE = 10

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# DATA FIELD MAPPINGS
# ============================================================================

# Map user-friendly names to actual column names if they differ
COLUMN_MAPPINGS = {
    'missing_visits': 'Missing Visits',
    'expected_visits': '# Expected Visits (Rave EDC : BO4)',
    'missing_pages': 'Missing Page',
    'total_pages': '# Pages Entered',
    'coded_terms': '# Coded terms',
    'uncoded_terms': '# Uncoded Terms',
    'open_queries': '#Total Queries',
    'closed_queries': '# DM Queries',  # Approximation
    'forms_requiring_verification': '# CRFs Require Verification (SDV)',
    'forms_verified': '# Forms Verified',
    'site_id': 'Site ID',
    'subject_id': 'Subject ID',
    'study_name': 'Project Name',
    'region': 'Region',
    'country': 'Country'
}

# ============================================================================
# AI INSIGHTS CONFIGURATION
# ============================================================================

# Prompts for different insight types
INSIGHT_PROMPTS = {
    'executive_summary': """
You are a Clinical Data Quality Expert analyzing pharmaceutical trial data.

STUDY DATA:
- Study ID: {study_id}
- Data Quality Index (DQI): {dqi_score}/100
- Risk Level: {risk_level}
- Sites: {site_count}
- Enrolled Subjects: {enrollment}

DETAILED METRICS:
- Missing Visits: {missing_visits_pct}%
- Missing Pages: {missing_pages_pct}%
- Open Queries: {open_queries_pct}%
- Unverified Forms: {unverified_forms_pct}%
- Uncoded Terms: {uncoded_terms_pct}%

Write a 2-3 sentence executive summary that:
1. Clearly states data quality status and risk
2. Identifies the critical blocking issue
3. Recommends immediate action

Use professional pharmaceutical language. Focus on URGENCY and CONCRETE NEXT STEPS.
Response: Just the summary, no preamble.""",

    'site_report': """
You are a CRA performance analyst.

SITE PERFORMANCE:
- Study: {study_id}
- Site ID: {site_id}
- CRA: {cra_name}
- Enrollment: {enrollment}/{target}
- Open Queries: {open_queries}
- Missing Pages: {missing_pages}
- Query Rate: {query_rate}%

Generate 3-4 bullet points assessing performance and listing specific action items.
Focus on what the CRA should do NEXT WEEK.
Response: Just bullet points, no preamble.""",

    'risk_alert': """
You are a Clinical Trial Risk Manager.

QUALITY TREND:
- Study: {study_id}
- Previous DQI: {dqi_previous}/100
- Current DQI: {dqi_current}/100
- Trend: {trend}
- Days to Interim Analysis: {days_to_interim}

Generate alert (2-3 sentences). If DQI declining AND days < 30: use 🚨 (URGENT).
Otherwise: use ⚠️ (STANDARD).

State: (1) The risk, (2) Timeline impact, (3) Escalation recommendation.
Response: Just the alert, no preamble."""
}

# ============================================================================
# BATCH PROCESSING CONFIGURATION
# ============================================================================

# Number of studies to process at once
BATCH_SIZE = 5

# Number of parallel workers for processing
PARALLEL_WORKERS = 4

# ============================================================================
# VALIDATION RULES
# ============================================================================

# Minimum data quality for analysis
MIN_DQI_FOR_INTERIM_ANALYSIS = 85
MIN_DQI_FOR_DATABASE_LOCK = 90

# Maximum acceptable percentages
MAX_MISSING_VISITS_PCT = 10
MAX_MISSING_PAGES_PCT = 15
MAX_OPEN_QUERIES_PCT = 20
MAX_UNCODED_TERMS_PCT = 5

# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Enable/disable features
FEATURES = {
    'ai_insights': True,
    'visualization': True,
    'risk_prediction': False,  # For Round 2
    'mobile_api': False,        # For Round 2
    'nlp_interface': False      # For Round 2
}

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

def print_config():
    """Print current configuration for debugging."""
    print("\n" + "="*80)
    print("NEST 2.0 CONFIGURATION")
    print("="*80)
    print(f"\nData Directories:")
    print(f"  Raw Data: {RAW_DATA_DIR}")
    print(f"  Processed Data: {PROCESSED_DATA_DIR}")
    print(f"  Outputs: {OUTPUT_DIR}")
    print(f"\nDQI Weights:")
    for component, weight in DQI_WEIGHTS.items():
        print(f"  {component.capitalize()}: {weight*100:.0f}%")
    print(f"\nRisk Thresholds:")
    for risk_level, (min_val, max_val) in RISK_THRESHOLDS.items():
        print(f"  {risk_level.upper()}: {min_val}-{max_val}")
    print(f"\nAPI Configuration:")
    print(f"  Model: {CLAUDE_MODEL}")
    print(f"  Max Tokens: {CLAUDE_MAX_TOKENS}")
    print(f"  API Key Configured: {'Yes' if ANTHROPIC_API_KEY else 'No'}")
    print("\n" + "="*80 + "\n")