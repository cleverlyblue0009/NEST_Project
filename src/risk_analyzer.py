# src/risk_analyzer.py
# Risk detection, scoring, and ranking for clinical trial studies

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from config import RISK_THRESHOLDS, RISK_LABELS, DQI_WEIGHTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskAnalyzer:
    """Analyze and rank studies by operational risk."""
    
    def __init__(self):
        self.risk_scores = {}
        self.ranked_studies = {}
    
    def analyze_study_risk(self, study_name: str, dqi_score: float, 
                          metrics: Dict) -> Dict:
        """
        Comprehensive risk analysis for a study.
        
        Args:
            study_name: Name of the study
            dqi_score: Calculated DQI (0-100)
            metrics: Dictionary with component percentages
            
        Returns:
            Risk analysis dictionary
        """
        
        risk_level = self._classify_risk(dqi_score)
        risk_score = self._calculate_risk_score(metrics)
        primary_driver = self._identify_primary_driver(metrics)
        secondary_drivers = self._identify_secondary_drivers(metrics)
        urgency = self._calculate_urgency(dqi_score, metrics)
        
        analysis = {
            'study_name': study_name,
            'dqi_score': dqi_score,
            'risk_level': risk_level,
            'risk_score': risk_score,
            'urgency': urgency,
            'primary_driver': primary_driver,
            'secondary_drivers': secondary_drivers,
            'metrics': metrics,
            'recommendations': self._generate_recommendations(dqi_score, metrics, primary_driver)
        }
        
        self.risk_scores[study_name] = analysis
        return analysis
    
    def _classify_risk(self, dqi_score: float) -> str:
        """Classify risk level based on DQI score."""
        if dqi_score < RISK_THRESHOLDS['high'][1]:  # < 70
            return 'HIGH RISK'
        elif dqi_score < RISK_THRESHOLDS['medium'][1]:  # < 80
            return 'MEDIUM RISK'
        else:
            return 'LOW RISK'
    
    def _calculate_risk_score(self, metrics: Dict) -> float:
        """
        Calculate overall risk score (opposite of DQI).
        Risk_Score = 100 - DQI
        """
        # Calculate impact-weighted risk
        components = [
            100 - metrics.get('missing_visits_pct', 0),
            100 - metrics.get('missing_pages_pct', 0),
            100 - metrics.get('open_queries_pct', 0),
            100 - metrics.get('unverified_forms_pct', 0),
            100 - metrics.get('uncoded_terms_pct', 0)
        ]
        weights = [0.25, 0.25, 0.20, 0.15, 0.15]
        
        # Risk is inverse of quality
        risk_score = 100 - sum(c * w for c, w in zip(components, weights))
        return max(0, min(100, risk_score))
    
    def _identify_primary_driver(self, metrics: Dict) -> str:
        """Identify the #1 quality issue dragging down the study."""
        impacts = {
            'Missing Visits': metrics.get('missing_visits_pct', 0) * DQI_WEIGHTS['visits'],
            'Missing Pages': metrics.get('missing_pages_pct', 0) * DQI_WEIGHTS['pages'],
            'Open Queries': metrics.get('open_queries_pct', 0) * DQI_WEIGHTS['queries'],
            'Unverified Forms': metrics.get('unverified_forms_pct', 0) * DQI_WEIGHTS['verification'],
            'Uncoded Terms': metrics.get('uncoded_terms_pct', 0) * DQI_WEIGHTS['safety']
        }
        
        return max(impacts, key=impacts.get)
    
    def _identify_secondary_drivers(self, metrics: Dict, top_n: int = 2) -> List[str]:
        """Identify the 2nd and 3rd biggest quality issues."""
        impacts = {
            'Missing Visits': metrics.get('missing_visits_pct', 0) * DQI_WEIGHTS['visits'],
            'Missing Pages': metrics.get('missing_pages_pct', 0) * DQI_WEIGHTS['pages'],
            'Open Queries': metrics.get('open_queries_pct', 0) * DQI_WEIGHTS['queries'],
            'Unverified Forms': metrics.get('unverified_forms_pct', 0) * DQI_WEIGHTS['verification'],
            'Uncoded Terms': metrics.get('uncoded_terms_pct', 0) * DQI_WEIGHTS['safety']
        }
        
        sorted_impacts = sorted(impacts.items(), key=lambda x: x[1], reverse=True)
        return [driver for driver, impact in sorted_impacts[1:top_n+1]]
    
    def _calculate_urgency(self, dqi_score: float, metrics: Dict) -> str:
        """
        Calculate urgency level for intervention.
        
        Urgency depends on:
        1. DQI score (how bad is it?)
        2. Trend (is it getting worse?)
        3. Specific metric thresholds
        """
        
        if dqi_score < 60:
            return 'CRITICAL - IMMEDIATE ACTION REQUIRED'
        elif dqi_score < 70:
            if metrics.get('missing_visits_pct', 0) > 15:
                return 'URGENT - ESCALATE WITHIN 24 HOURS'
            elif metrics.get('open_queries_pct', 0) > 25:
                return 'URGENT - ESCALATE WITHIN 24 HOURS'
            else:
                return 'HIGH - ESCALATE WITHIN 48 HOURS'
        elif dqi_score < 80:
            if metrics.get('missing_visits_pct', 0) > 10:
                return 'MEDIUM - MONITOR CLOSELY'
            else:
                return 'MEDIUM - STANDARD MONITORING'
        else:
            return 'LOW - ROUTINE MONITORING'
    
    def _generate_recommendations(self, dqi_score: float, metrics: Dict, 
                                 primary_driver: str) -> List[str]:
        """Generate specific action recommendations based on analysis."""
        recommendations = []
        
        # Based on primary driver
        if primary_driver == 'Missing Visits':
            recommendations.append("Deploy CRA to sites with highest missing visit counts")
            recommendations.append("Conduct telephone follow-ups with subjects for overdue visits")
            recommendations.append("Review visit scheduling procedures at underperforming sites")
        
        elif primary_driver == 'Missing Pages':
            recommendations.append("Implement CRF entry refresher training at sites")
            recommendations.append("Escalate to data entry teams for accelerated page completion")
            recommendations.append("Identify if certain form types are consistently problematic")
        
        elif primary_driver == 'Open Queries':
            recommendations.append("Prioritize query resolution - target closure within 7 days")
            recommendations.append("Escalate high-priority queries to Site Principal Investigator")
            recommendations.append("Consider query amnesty program to clear backlog")
        
        elif primary_driver == 'Unverified Forms':
            recommendations.append("Implement SDV for all outstanding forms")
            recommendations.append("Allocate additional verification resources")
            recommendations.append("Schedule verification activities within 5 days")
        
        elif primary_driver == 'Uncoded Terms':
            recommendations.append("Engage coding team for accelerated coding cycle")
            recommendations.append("Prioritize safety-critical terms for coding")
            recommendations.append("Consider external coding vendor support if backlog > 100 terms")
        
        # General recommendations based on DQI
        if dqi_score < 70:
            recommendations.append("Schedule urgent study status meeting with Clinical Project Manager")
            recommendations.append("Implement daily monitoring of quality metrics")
        
        if metrics.get('missing_visits_pct', 0) > 15 or metrics.get('open_queries_pct', 0) > 25:
            recommendations.append("Assess risk to database lock timeline and interim analysis readiness")
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def rank_studies(self, studies_analysis: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Rank all studies by risk level.
        
        Args:
            studies_analysis: Dictionary of study analyses
            
        Returns:
            Ranked dictionary (HIGH RISK first)
        """
        
        # Sort by DQI score (lowest = highest risk)
        ranked = {}
        sorted_studies = sorted(studies_analysis.items(), 
                               key=lambda x: x[1]['dqi_score'])
        
        for rank, (study_name, analysis) in enumerate(sorted_studies, 1):
            analysis['rank'] = rank
            ranked[study_name] = analysis
        
        self.ranked_studies = ranked
        return ranked
    
    def get_high_risk_studies(self, threshold: float = 70) -> Dict[str, Dict]:
        """Get all studies with DQI below threshold."""
        high_risk = {name: analysis for name, analysis in self.ranked_studies.items()
                    if analysis['dqi_score'] < threshold}
        return high_risk
    
    def get_top_n_studies(self, n: int = 10) -> Dict[str, Dict]:
        """Get top N highest-risk studies."""
        return dict(list(self.ranked_studies.items())[:n])
    
    def get_risk_summary(self) -> Dict:
        """Get summary of all studies by risk category."""
        summary = {
            'HIGH RISK': 0,
            'MEDIUM RISK': 0,
            'LOW RISK': 0,
            'average_dqi': 0,
            'median_dqi': 0,
            'studies_by_risk': {
                'HIGH RISK': [],
                'MEDIUM RISK': [],
                'LOW RISK': []
            }
        }
        
        all_dqi = []
        for study_name, analysis in self.ranked_studies.items():
            risk_level = analysis['risk_level']
            dqi_score = analysis['dqi_score']
            
            summary[risk_level] += 1
            summary['studies_by_risk'][risk_level].append({
                'study': study_name,
                'dqi': dqi_score,
                'driver': analysis['primary_driver']
            })
            all_dqi.append(dqi_score)
        
        if all_dqi:
            summary['average_dqi'] = np.mean(all_dqi)
            summary['median_dqi'] = np.median(all_dqi)
        
        return summary
    
    def identify_anomalies(self, threshold_std: float = 2.0) -> Dict[str, str]:
        """
        Identify statistical anomalies (outliers).
        Studies with DQI > 2 std deviations from mean.
        """
        dqi_scores = [a['dqi_score'] for a in self.ranked_studies.values()]
        mean_dqi = np.mean(dqi_scores)
        std_dqi = np.std(dqi_scores)
        
        anomalies = {}
        for study_name, analysis in self.ranked_studies.items():
            dqi = analysis['dqi_score']
            z_score = abs((dqi - mean_dqi) / std_dqi) if std_dqi > 0 else 0
            
            if z_score > threshold_std:
                if dqi < mean_dqi:
                    anomalies[study_name] = f"Significantly BELOW average (Z={z_score:.2f})"
                else:
                    anomalies[study_name] = f"Significantly ABOVE average (Z={z_score:.2f})"
        
        return anomalies


# USAGE EXAMPLE
if __name__ == '__main__':
    analyzer = RiskAnalyzer()
    
    # Example: Analyze a study
    example_metrics = {
        'missing_visits_pct': 12,
        'missing_pages_pct': 15,
        'open_queries_pct': 22.5,
        'unverified_forms_pct': 25,
        'uncoded_terms_pct': 15
    }
    
    analysis = analyzer.analyze_study_risk(
        study_name='Study_0456',
        dqi_score=82.8,
        metrics=example_metrics
    )
    
    print("\n" + "="*70)
    print("EXAMPLE RISK ANALYSIS")
    print("="*70)
    print(f"Study: {analysis['study_name']}")
    print(f"DQI Score: {analysis['dqi_score']:.1f}")
    print(f"Risk Level: {analysis['risk_level']}")
    print(f"Urgency: {analysis['urgency']}")
    print(f"Primary Driver: {analysis['primary_driver']}")
    print(f"\nRecommendations:")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"  {i}. {rec}")
    print("\n" + "="*70)