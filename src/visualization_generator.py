# src/visualization_generator.py
# Professional Scientific Visualizations for NEST 2.0

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
from typing import Dict

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class VisualizationGenerator:
    """Generate publication-quality figures for NEST 2.0."""
    
    def __init__(self, output_dir='./data/outputs/visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = 300  # High resolution for presentations
    
    def plot_dqi_distribution(self, dqi_scores: Dict[str, float]):
        """
        Plot 1: DQI Distribution across all 25 studies
        Shows the landscape: how many studies are at risk?
        """
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        
        scores = list(dqi_scores.values())
        
        # Create histogram
        n, bins, patches = ax.hist(scores, bins=15, edgecolor='black', alpha=0.7)
        
        # Color code by risk level
        for i, patch in enumerate(patches):
            if bins[i] < 70:
                patch.set_facecolor('#EF553B')  # Red - HIGH RISK
            elif bins[i] < 80:
                patch.set_facecolor('#FFA15A')  # Orange - MEDIUM RISK
            else:
                patch.set_facecolor('#00CC96')  # Green - LOW RISK
        
        # Add reference lines
        ax.axvline(70, color='red', linestyle='--', linewidth=2, alpha=0.7, label='High Risk Threshold')
        ax.axvline(80, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Medium Risk Threshold')
        ax.axvline(np.mean(scores), color='blue', linestyle='-', linewidth=2.5, label=f'Mean DQI: {np.mean(scores):.1f}')
        
        ax.set_xlabel('Data Quality Index (DQI) Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Studies', fontsize=12, fontweight='bold')
        ax.set_title('Clinical Trial DQI Distribution Across 25 Studies', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dqi_distribution.png', dpi=self.dpi, bbox_inches='tight')
        print("✓ Saved: dqi_distribution.png")
        plt.close()
    
    def plot_risk_heatmap(self, risk_matrix: pd.DataFrame):
        """
        Plot 2: Risk Heatmap
        Studies (rows) vs. Quality Drivers (columns)
        Shows which studies are failing on which metrics.
        """
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
        
        # Create heatmap
        sns.heatmap(risk_matrix, 
                   annot=True, 
                   fmt='.1f',
                   cmap='RdYlGn_r',  # Red=bad, Green=good
                   cbar_kws={'label': 'Risk Score (0-100)'},
                   linewidths=0.5,
                   linecolor='gray',
                   ax=ax,
                   vmin=0,
                   vmax=100)
        
        ax.set_xlabel('Data Quality Dimensions', fontsize=12, fontweight='bold')
        ax.set_ylabel('Study ID (Top 15 At-Risk)', fontsize=12, fontweight='bold')
        ax.set_title('Risk Heatmap: Quality Drivers by Study', 
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_heatmap.png', dpi=self.dpi, bbox_inches='tight')
        print("✓ Saved: risk_heatmap.png")
        plt.close()
    
    def plot_study_comparison(self, top_studies: Dict):
        """
        Plot 3: Top 10 At-Risk Studies Comparison
        Shows DQI and component breakdown for each study.
        """
        study_ids = list(top_studies.keys())[:10]
        dqi_scores = [top_studies[s]['dqi_score'] for s in study_ids]
        risk_levels = [top_studies[s]['risk_level'] for s in study_ids]
        
        # Color by risk
        colors = ['#EF553B' if level == 'HIGH RISK' else '#FFA15A' 
                  for level in risk_levels]
        
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
        
        x_pos = np.arange(len(study_ids))
        bars = ax.barh(x_pos, dqi_scores, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, dqi_scores)):
            ax.text(score + 1, i, f'{score:.1f}', va='center', fontweight='bold')
        
        # Add risk threshold line
        ax.axvline(70, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Risk Threshold (70)')
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels(study_ids, fontsize=10)
        ax.set_xlabel('DQI Score (0-100)', fontsize=12, fontweight='bold')
        ax.set_title('Top 10 High-Risk Studies - DQI Comparison', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 105)
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'study_comparison.png', dpi=self.dpi, bbox_inches='tight')
        print("✓ Saved: study_comparison.png")
        plt.close()
    
    def plot_dqi_components(self, study_id: str, metrics: Dict):
        """
        Plot 4: DQI Component Breakdown (Radar/Pie Chart)
        Shows how each quality dimension contributes to overall DQI.
        """
        # Extract metrics
        categories = ['Visit\nCompleteness', 'Form Data\nQuality', 
                     'Query\nResolution', 'Form\nVerification', 'Safety/Coding\nCompleteness']
        
        # Calculate component scores (higher is better)
        component_scores = [
            100 - metrics['missing_visits_pct'],
            100 - metrics['missing_pages_pct'],
            100 - metrics['open_queries_pct'],
            100 - metrics['unverified_forms_pct'],
            100 - metrics['uncoded_terms_pct']
        ]
        
        weights = [0.25, 0.25, 0.20, 0.15, 0.15]
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
        
        # LEFT: Radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        component_scores_plot = component_scores + [component_scores[0]]
        angles_plot = angles + [angles[0]]
        
        ax1 = plt.subplot(121, projection='polar')
        ax1.plot(angles_plot, component_scores_plot, 'o-', linewidth=2.5, 
                color='#636EFA', markersize=8)
        ax1.fill(angles_plot, component_scores_plot, alpha=0.25, color='#636EFA')
        ax1.set_xticks(angles)
        ax1.set_xticklabels(categories, fontsize=9)
        ax1.set_ylim(0, 100)
        ax1.set_title(f'{study_id}\nData Quality Components', 
                     fontsize=12, fontweight='bold', pad=20)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # RIGHT: Weighted contribution bar chart
        ax2 = plt.subplot(122)
        weighted_impact = [score * weight for score, weight in zip(component_scores, weights)]
        colors_bar = plt.cm.RdYlGn(np.array(component_scores) / 100)
        
        bars = ax2.barh(categories, weighted_impact, color=colors_bar, edgecolor='black', linewidth=1.5)
        
        for bar, impact in zip(bars, weighted_impact):
            ax2.text(impact + 0.3, bar.get_y() + bar.get_height()/2, 
                    f'{impact:.2f}', va='center', fontweight='bold', fontsize=10)
        
        ax2.set_xlabel('Weighted Contribution to DQI', fontsize=11, fontweight='bold')
        ax2.set_title('Impact on Overall DQI Score', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, max(weighted_impact) + 2)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'dqi_components_{study_id}.png', dpi=self.dpi, bbox_inches='tight')
        print(f"✓ Saved: dqi_components_{study_id}.png")
        plt.close()
    
    def plot_dqi_formula_infographic(self):
        """
        Plot 5: DQI Formula Infographic
        Visual representation of how DQI is calculated.
        Perfect for presentations - judges love this.
        """
        fig, ax = plt.subplots(figsize=(14, 8), facecolor='#f8f9fa')
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Data Quality Index (DQI) Calculation', 
               ha='center', va='top', fontsize=16, fontweight='bold', 
               transform=ax.transAxes)
        
        # Formula components
        components = [
            ('Visit\nCompleteness', '25%', '#FF6B6B'),
            ('Form Data\nQuality', '25%', '#4ECDC4'),
            ('Query\nResolution', '20%', '#45B7D1'),
            ('Form\nVerification', '15%', '#FFA07A'),
            ('Safety/Coding\nCompleteness', '15%', '#98D8C8')
        ]
        
        y_start = 0.75
        box_height = 0.12
        box_width = 0.18
        
        boxes = []
        for i, (name, weight, color) in enumerate(components):
            x = 0.05 + i * 0.19
            
            # Draw box
            fancy_box = FancyBboxPatch((x, y_start - box_height), box_width, box_height,
                                      boxstyle="round,pad=0.01", 
                                      transform=ax.transAxes,
                                      facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
            ax.add_patch(fancy_box)
            
            # Add text
            ax.text(x + box_width/2, y_start - box_height/3, name,
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   transform=ax.transAxes, color='white')
            ax.text(x + box_width/2, y_start - box_height*0.8, weight,
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   transform=ax.transAxes, color='white')
            
            # Draw arrow
            if i < len(components) - 1:
                ax.annotate('', xy=(x + box_width + 0.005, y_start - box_height/2),
                           xytext=(x + box_width - 0.005, y_start - box_height/2),
                           xycoords='axes fraction',
                           arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
        
        # Formula equation
        ax.text(0.5, 0.50, 'DQI = 0.25(Visit%) + 0.25(Form%) + 0.20(Query%) + 0.15(Verification%) + 0.15(Safety%)',
               ha='center', va='center', fontsize=13, fontweight='bold',
               transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='#FFF9E6', edgecolor='black', linewidth=2, pad=1))
        
        # Risk thresholds
        threshold_y = 0.30
        ax.text(0.5, threshold_y + 0.08, 'Risk Classification Thresholds',
               ha='center', va='top', fontsize=12, fontweight='bold',
               transform=ax.transAxes)
        
        thresholds = [
            ('HIGH RISK', 'DQI < 70', '#EF553B'),
            ('MEDIUM RISK', '70 ≤ DQI < 80', '#FFA15A'),
            ('LOW RISK', 'DQI ≥ 80', '#00CC96')
        ]
        
        for i, (label, range_text, color) in enumerate(thresholds):
            x = 0.2 + i * 0.3
            fancy_box = FancyBboxPatch((x - 0.08, threshold_y - 0.05), 0.16, 0.06,
                                      boxstyle="round,pad=0.01",
                                      transform=ax.transAxes,
                                      facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.8)
            ax.add_patch(fancy_box)
            
            ax.text(x, threshold_y - 0.01, label, ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white', transform=ax.transAxes)
            ax.text(x, threshold_y - 0.04, range_text, ha='center', va='center',
                   fontsize=9, color='white', transform=ax.transAxes)
        
        # Key insight
        ax.text(0.5, 0.05, 'The DQI metric unifies 5 critical quality dimensions into a single actionable score (0-100)',
               ha='center', va='bottom', fontsize=11, style='italic', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='#E8F4F8', alpha=0.8, pad=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dqi_formula_infographic.png', dpi=self.dpi, bbox_inches='tight')
        print("✓ Saved: dqi_formula_infographic.png")
        plt.close()
    
    def generate_all_visualizations(self, dqi_results_file: str):
        """Master function: Generate all visualizations from DQI results."""
        
        with open(dqi_results_file, 'r') as f:
            results = json.load(f)
        
        # Extract data
        dqi_scores = {study: data['study_level']['dqi_score'] 
                     for study, data in results.items()}
        
        print("\n" + "="*70)
        print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
        print("="*70 + "\n")
        
        # Plot 1: DQI Distribution
        self.plot_dqi_distribution(dqi_scores)
        
        # Plot 2: Risk Heatmap (top 15 studies × 5 metrics)
        top_studies = {k: results[k]['study_level'] 
                      for k in list(dqi_scores.keys())[:15]}
        
        risk_data = []
        for study_id, metrics in top_studies.items():
            risk_data.append({
                'Study': study_id,
                'Missing Visits': metrics['metrics']['missing_visits_pct'],
                'Missing Pages': metrics['metrics']['missing_pages_pct'],
                'Open Queries': metrics['metrics']['open_queries_pct'],
                'Unverified Forms': metrics['metrics']['unverified_forms_pct'],
                'Uncoded Terms': metrics['metrics']['uncoded_terms_pct']
            })
        
        risk_df = pd.DataFrame(risk_data).set_index('Study')
        self.plot_risk_heatmap(risk_df)
        
        # Plot 3: Study Comparison
        top_10 = {k: results[k]['study_level'] 
                 for k in list(dqi_scores.keys())[:10]}
        self.plot_study_comparison(top_10)
        
        # Plot 4: DQI Components for top 3 studies
        for study_id in list(dqi_scores.keys())[:3]:
            metrics = results[study_id]['study_level']['metrics']
            self.plot_dqi_components(study_id, metrics)
        
        # Plot 5: Formula Infographic
        self.plot_dqi_formula_infographic()
        
        print("\n✓ All visualizations generated successfully!")
        print(f"✓ Saved to: {self.output_dir}")


# USAGE
if __name__ == '__main__':
    viz_gen = VisualizationGenerator()
    viz_gen.generate_all_visualizations('./data/processed/dqi_scores.json')