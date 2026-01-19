# src/__init__.py
# NEST 2.0 Package Initialization

__version__ = '2.0.0'
__author__ = 'Clinical Data Intelligence Team'
__description__ = 'NEST 2.0: Real-Time Operational Dataflow Intelligence for Clinical Trials'

# Import main modules for easy access
from .config import (
    DQI_WEIGHTS, RISK_THRESHOLDS, RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, OUTPUT_DIR, ANTHROPIC_API_KEY
)
from .data_loader import DataLoader
from .metrics_calculator import DQICalculator, BatchDQIProcessor
from .risk_analyzer import RiskAnalyzer
from .ai_insights_generator import AIInsightsGenerator, BatchInsightsProcessor
from .visualization_generator import VisualizationGenerator

__all__ = [
    'DataLoader',
    'DQICalculator',
    'BatchDQIProcessor',
    'RiskAnalyzer',
    'AIInsightsGenerator',
    'BatchInsightsProcessor',
    'VisualizationGenerator',
    'DQI_WEIGHTS',
    'RISK_THRESHOLDS',
    'RAW_DATA_DIR',
    'PROCESSED_DATA_DIR',
    'OUTPUT_DIR'
]

print(f"✓ NEST 2.0 v{__version__} loaded successfully")
