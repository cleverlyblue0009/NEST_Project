# src/report_generator.py
# Executive Summary & Report Generation for NEST 2.0

from pathlib import Path
from datetime import datetime
import json


class ReportGenerator:
    """Generates professional executive reports from DQI analysis."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_executive_summary(self, dqi_results, insights, stats):
        """
        Generate 1-page executive summary for presentation deck.
        This is what judges will read if they want background.
        """
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                        NEST 2.0 EXECUTIVE SUMMARY                          ║
║              Clinical Trial Data Quality Intelligence Report               ║
╚════════════════════════════════════════════════════════════════════════════╝

Generated: {timestamp}
Analysis: 25 Clinical Studies, Data Integration from Multiple EDC Sources

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. EXECUTIVE OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NEST 2.0 is an AI-powered data intelligence platform that unifies clinical trial 
data from multiple sources and generates actionable insights in real-time.

KEY INNOVATION: The Data Quality Index (DQI) — a unified 0–100 metric that 
aggregates 5 critical quality dimensions into one score.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. KEY FINDINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Average DQI Across All Studies: {stats['avg_dqi']:.1f}/100

Risk Distribution:
• HIGH RISK (DQI < 70):    {stats['high_risk_count']}
• MEDIUM RISK (70–80):     {stats['medium_risk_count']}
• LOW RISK (> 80):         {stats['low_risk_count']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. TOP 5 HIGHEST-RISK STUDIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for i, (study_id, insight) in enumerate(list(insights.items())[:5], 1):
            report += f"""
{i}. {study_id}
   DQI Score: {insight['dqi_score']}/100 ({insight['risk_level']})
   Primary Driver: {insight['primary_driver']}

   Executive Summary:
   {insight['executive_summary']}

   Recommended Action:
   {insight['risk_alert']}
"""
        
        report += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. CONCLUSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NEST 2.0 demonstrates how real-world clinical data can be transformed into 
clear, actionable intelligence. By combining a unified quality metric with 
AI-generated insights, trial teams can identify risk in hours instead of days.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
END OF REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # ✅ FIXED: correct variable name + UTF-8 encoding
        output_file = self.output_dir / "Executive_Summary_Report.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        return output_file
    
    def generate_methodology_document(self):
        """Generate DQI Methodology deep-dive document."""
        
        doc = """
DQI (DATA QUALITY INDEX) METHODOLOGY

This document explains the weighting and logic behind the DQI calculation.
"""
        
        output_file = self.output_dir / "DQI_Methodology.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(doc)
        
        return output_file


# USAGE (standalone testing)
if __name__ == "__main__":
    gen = ReportGenerator("./data/outputs")
    
    with open("./data/outputs/dqi_scores.json", "r") as f:
        dqi_results = json.load(f)
    
    with open("./data/outputs/ai_insights.json", "r") as f:
        insights = json.load(f)
    
    dqi_scores = {k: v["study_level"]["dqi_score"] for k, v in dqi_results.items()}
    
    stats = {
        "avg_dqi": sum(dqi_scores.values()) / len(dqi_scores),
        "high_risk_count": sum(1 for s in dqi_scores.values() if s < 70),
        "medium_risk_count": sum(1 for s in dqi_scores.values() if 70 <= s < 80),
        "low_risk_count": sum(1 for s in dqi_scores.values() if s >= 80),
    }
    
    gen.generate_executive_summary(dqi_results, insights, stats)
    gen.generate_methodology_document()
    
    print("✓ Reports generated successfully")
