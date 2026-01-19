# scripts/run_full_pipeline.py
# Master script - Orchestrates entire NEST 2.0 analysis pipeline

import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from metrics_calculator import BatchDQIProcessor
from ai_insights_generator import BatchInsightsProcessor
from visualization_generator import VisualizationGenerator
from report_generator import ReportGenerator


class NESTPipeline:
    """Master orchestrator for NEST 2.0 analysis."""
    
    def __init__(self, data_dir='./data/raw', output_dir='./data/outputs'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run(self):
        print("\n" + "=" * 80)
        print(" " * 20 + "NEST 2.0 CLINICAL TRIAL DATA INTELLIGENCE")
        print(" " * 25 + "Full Pipeline Execution")
        print("=" * 80)
        
        # ------------------------------------------------------------
        # PHASE 1: DQI CALCULATION
        # ------------------------------------------------------------
        print("\n[PHASE 1] Computing DQI Scores...")
        print("-" * 80)
        
        dqi_processor = BatchDQIProcessor()
        dqi_results = dqi_processor.process_all_studies(str(self.data_dir))
        
        dqi_file = self.output_dir / "dqi_scores.json"
        dqi_processor.export_results(str(dqi_file), dqi_results)
        
        # Summary stats (CORRECT SCHEMA)
        dqi_scores = {
            study_id: data["study_level"]["dqi_score"]
            for study_id, data in dqi_results.items()
        }
        
        avg_dqi = sum(dqi_scores.values()) / len(dqi_scores)
        high_risk_count = sum(1 for v in dqi_scores.values() if v < 70)
        medium_risk_count = sum(1 for v in dqi_scores.values() if 70 <= v < 80)
        low_risk_count = sum(1 for v in dqi_scores.values() if v >= 80)
        
        print(f"\n✓ Processed {len(dqi_scores)} studies")
        print(f"  • Average DQI: {avg_dqi:.2f}/100")
        print(f"  • High Risk: {high_risk_count}")
        print(f"  • Medium Risk: {medium_risk_count}")
        print(f"  • Low Risk: {low_risk_count}")
        
        # ------------------------------------------------------------
        # PHASE 2: AI INSIGHTS (PASS STRUCTURE AS-IS)
        # ------------------------------------------------------------
        print("\n[PHASE 2] Generating AI Insights...")
        print("-" * 80)
        
        insights_processor = BatchInsightsProcessor()
        insights = insights_processor.generator.generate_batch_insights(dqi_results)
        
        insights_file = self.output_dir / "ai_insights.json"
        insights_processor.generator.export_insights(insights, str(insights_file))
        
        print(f"\n✓ Generated insights for {len(insights)} studies")
        
        # Sample
        for study_id, insight in list(insights.items())[:2]:
            print(f"\n📊 {study_id}")
            print(f"   DQI: {insight['dqi_score']} ({insight['risk_level']})")
            print(f"   Driver: {insight['primary_driver']}")
            print(f"   Summary: {insight['executive_summary']}")
        
        # ------------------------------------------------------------
        # PHASE 3: VISUALIZATIONS
        # ------------------------------------------------------------
        print("\n[PHASE 3] Creating Visualizations...")
        print("-" * 80)
        
        viz_gen = VisualizationGenerator(str(self.output_dir / "visualizations"))
        viz_gen.generate_all_visualizations(str(dqi_file))
        
        # ------------------------------------------------------------
        # PHASE 4: REPORT
        # ------------------------------------------------------------
        print("\n[PHASE 4] Generating Executive Report...")
        print("-" * 80)
        
        report_gen = ReportGenerator(self.output_dir)
        report_file = report_gen.generate_executive_summary(
            dqi_results=dqi_results,
            insights=insights,
            stats={
                "avg_dqi": avg_dqi,
                "high_risk_count": high_risk_count,
                "medium_risk_count": medium_risk_count,
                "low_risk_count": low_risk_count
            }
        )
        
        print(f"\n✓ Report saved: {report_file}")
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE")
        print("=" * 80)
        
        return {
            "dqi_results": dqi_results,
            "insights": insights,
            "statistics": {
                "avg_dqi": avg_dqi,
                "high_risk": high_risk_count,
                "medium_risk": medium_risk_count,
                "low_risk": low_risk_count
            }
        }


if __name__ == "__main__":
    pipeline = NESTPipeline(
        data_dir="./data/raw",
        output_dir="./data/outputs"
    )
    
    results = pipeline.run()
    
    with open("./data/outputs/pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
