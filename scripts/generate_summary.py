"""
STEP 5: EXECUTIVE SUMMARY & RECOMMENDATIONS
==========================================
Problem Statement Mapping:
- Provides "actionable insights" for CTTs & CRAs in clear, decision-ready format
- Enables "intelligent collaboration" by surfacing specific, templated recommendations
- No LLM API calls; uses rule-based templates for deterministic, explainable output

What This Does:
1. Reads risk rankings from STEP 4
2. Generates short, templated executive summaries:
   - One-liner for each study (problem statement + driver)
   - Targeted recommendations based on risk level & top drivers
3. Produces a consolidated executive summary suitable for:
   - PowerPoint presentations
   - Clinical trial team briefings
   - CRA action items
4. NO LLM/API calls—all logic is explicit and auditable

How It Connects:
- INPUT: risk_rankings.csv + risk_rankings_site_level.csv from STEP 4
- OUTPUT: executive_summary.txt (ready for presentation to judges & stakeholders)
- This is STEP 5 of the 5-step pipeline (final step)
"""

import os
import pandas as pd

def generate_study_recommendation(study_id, dqi_score, risk_level, risk_drivers, signals_dict):
    """
    Generate templated recommendation for a single study.
    
    Args:
        study_id (str): Study identifier
        dqi_score (float): DQI score
        risk_level (str): Risk category
        risk_drivers (str): Top risk drivers (comma-separated)
        signals_dict (dict): Signal counts {signal_type: count}
    
    Returns:
        str: Formatted recommendation
    """
    
    # Problem statement template
    problem = f"{study_id} is {risk_level.lower()}"
    
    # Add specific drivers
    if risk_drivers:
        problem += f" due to {risk_drivers.lower()}"
    else:
        problem += " — review immediately"
    
    # Actionable recommendations based on risk level
    if risk_level == "High Risk":
        actions = [
            "• IMMEDIATE: Convene study safety/monitoring committee",
            "• Conduct focused data audit on highest-impact issues",
            "• Assign dedicated CRA resources for remediation",
            "• Create 48-hour corrective action plan",
            "• Escalate to Clinical Trial Team Lead"
        ]
    elif risk_level == "Medium Risk":
        actions = [
            "• Schedule weekly monitoring calls with site",
            "• Develop corrective action plan (target: 2 weeks)",
            "• Assign CRA to verify remediation",
            "• Review data quality metrics bi-weekly",
            "• Document improvements in study binder"
        ]
    else:  # Low Risk
        actions = [
            "• Continue routine monitoring",
            "• Verify issue resolution at next site visit",
            "• Monitor for any trend toward higher risk",
            "• Update study status in trial management system"
        ]
    
    # Specific guidance based on top drivers
    specific_guidance = ""
    if "missing visits" in risk_drivers.lower():
        specific_guidance += "\n  → Missing Visit Action: Confirm expected visit schedule with site; verify patient completion status"
    if "edrr" in risk_drivers.lower():
        specific_guidance += "\n  → Query Action: Accelerate resolution; prioritize critical data elements"
    if "uncoded" in risk_drivers.lower():
        specific_guidance += "\n  → Coding Action: Review coding manual with site; consider coding service support"
    if "sae" in risk_drivers.lower():
        specific_guidance += "\n  → SAE Action: Expedite clinical assessment; verify regulatory reporting timeline"
    if "pages" in risk_drivers.lower():
        specific_guidance += "\n  → CRF Action: Confirm form submission; address technical/process barriers"
    
    # Format output
    output = f"\n{'='*80}\nSTUDY: {study_id}\n{'='*80}"
    output += f"\n\nData Quality Index (DQI): {dqi_score:.2f} / 100"
    output += f"\nRisk Level: {risk_level}"
    output += f"\nTop Risk Drivers: {risk_drivers if risk_drivers else 'Multiple issues'}"
    output += f"\n\nProblem Statement:\n  {problem}."
    output += f"\n\nRecommended Actions:"
    for action in actions:
        output += f"\n{action}"
    
    if specific_guidance:
        output += f"\n\nSpecific Guidance:{specific_guidance}"
    
    # Summary of signals
    output += f"\n\nSignal Summary:"
    for signal_type, count in signals_dict.items():
        if count > 0:
            output += f"\n  • {signal_type}: {int(count)}"
    
    return output


def generate_site_recommendation(study_id, site_id, dqi_score, risk_level, risk_drivers):
    """
    Generate brief site-level recommendation.
    
    Args:
        study_id (str): Study identifier
        site_id (str): Site identifier
        dqi_score (float): DQI score
        risk_level (str): Risk category
        risk_drivers (str): Top risk drivers
    
    Returns:
        str: Brief site recommendation
    """
    
    if risk_level == "High Risk":
        action = "URGENT: On-site data audit required"
    elif risk_level == "Medium Risk":
        action = "Schedule enhanced site monitoring"
    else:
        action = "Routine monitoring"
    
    return f"  {study_id} - {site_id}: DQI {dqi_score:.1f} ({risk_level}) | {action}"


def generate_executive_summary(study_ranks, site_ranks, output_path="outputs/executive_summary.txt"):
    """
    Generate comprehensive executive summary with study-level and site-level details.
    
    Args:
        study_ranks (pd.DataFrame): Ranked studies from STEP 4
        site_ranks (pd.DataFrame): Ranked sites from STEP 4
        output_path (str): Where to save summary
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("CLINICAL TRIAL DATA QUALITY & OPERATIONAL RISK INTELLIGENCE REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write("System: Automated Clinical Data Quality & Operational Risk Intelligence\n\n")
        
        # Executive Summary
        high_risk_count = len(study_ranks[study_ranks['risk_level'] == 'High Risk'])
        medium_risk_count = len(study_ranks[study_ranks['risk_level'] == 'Medium Risk'])
        low_risk_count = len(study_ranks[study_ranks['risk_level'] == 'Low Risk'])
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Studies Analyzed: {len(study_ranks)}\n")
        f.write(f"High Risk Studies: {high_risk_count}\n")
        f.write(f"Medium Risk Studies: {medium_risk_count}\n")
        f.write(f"Low Risk Studies: {low_risk_count}\n")
        f.write(f"Average DQI Across All Studies: {study_ranks['dqi_score'].mean():.2f} / 100\n\n")
        
        # Key Findings
        f.write("KEY FINDINGS\n")
        f.write("-" * 80 + "\n")
        if high_risk_count > 0:
            f.write(f"⚠ {high_risk_count} study(ies) require immediate attention due to critical data quality issues.\n")
        if medium_risk_count > 0:
            f.write(f"→ {medium_risk_count} study(ies) need structured monitoring and corrective actions.\n")
        if low_risk_count > 0:
            f.write(f"✓ {low_risk_count} study(ies) are at low risk; routine monitoring recommended.\n")
        f.write("\n")
        
        # Overall risk drivers
        f.write("TOP OPERATIONAL RISK DRIVERS (ACROSS ALL STUDIES)\n")
        f.write("-" * 80 + "\n")
        
        total_issues = {
            'missing_pages': study_ranks['missing_pages'].sum(),
            'missing_visits': study_ranks['missing_visits'].sum(),
            'unresolved_edrr': study_ranks['unresolved_edrr'].sum(),
            'uncoded_terms': study_ranks['uncoded_terms'].sum(),
            'pending_sae_reviews': study_ranks['pending_sae_reviews'].sum(),
        }
        
        sorted_issues = sorted(total_issues.items(), key=lambda x: x[1], reverse=True)
        for issue_type, count in sorted_issues:
            if count > 0:
                display_name = {
                    'missing_pages': 'CRF Pages Incomplete',
                    'missing_visits': 'Missing Visits',
                    'unresolved_edrr': 'Unresolved Queries (EDRR)',
                    'uncoded_terms': 'Uncoded Medical Terms',
                    'pending_sae_reviews': 'Pending SAE Reviews',
                }
                f.write(f"  • {display_name[issue_type]}: {int(count)} instances\n")
        f.write("\n")
        
        # Study-level recommendations
        f.write("=" * 80 + "\n")
        f.write("STUDY-LEVEL RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n")
        
        for idx, row in study_ranks.iterrows():
            signals_dict = {
                'CRF Pages': row['missing_pages'],
                'Missing Visits': row['missing_visits'],
                'EDRR Queries': row['unresolved_edrr'],
                'Uncoded Terms': row['uncoded_terms'],
                'SAE Reviews': row['pending_sae_reviews'],
            }
            
            recommendation = generate_study_recommendation(
                study_id=row['study_id'],
                dqi_score=row['dqi_score'],
                risk_level=row['risk_level'],
                risk_drivers=row['top_risk_drivers'],
                signals_dict=signals_dict
            )
            f.write(recommendation)
        
        f.write("\n\n")
        
        # Site-level summary
        f.write("=" * 80 + "\n")
        f.write("SITE-LEVEL RISK SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Sites Analyzed: {len(site_ranks)}\n")
        f.write(f"High Risk Sites: {len(site_ranks[site_ranks['risk_level'] == 'High Risk'])}\n")
        f.write(f"Medium Risk Sites: {len(site_ranks[site_ranks['risk_level'] == 'Medium Risk'])}\n")
        f.write(f"Low Risk Sites: {len(site_ranks[site_ranks['risk_level'] == 'Low Risk'])}\n\n")
        
        f.write("Top 10 Most At-Risk Sites (Global Ranking):\n")
        f.write("-" * 80 + "\n")
        for idx, row in site_ranks.head(10).iterrows():
            f.write(generate_site_recommendation(
                study_id=row['study_id'],
                site_id=row['site_id'],
                dqi_score=row['dqi_score'],
                risk_level=row['risk_level'],
                risk_drivers=row['top_risk_drivers']
            ) + "\n")
        
        f.write("\n\n")
        
        # Closing recommendations
        f.write("=" * 80 + "\n")
        f.write("OPERATIONAL RECOMMENDATIONS FOR CLINICAL TRIAL TEAMS\n")
        f.write("=" * 80 + "\n")
        
        recommendations = [
            "1. IMMEDIATE (Next 48 hours):",
            "   • Escalate all High Risk studies to Clinical Trial Lead",
            "   • Initiate focused data audits at High Risk sites",
            "   • Convene study safety committee for High Risk studies",
            "",
            "2. SHORT-TERM (1-2 weeks):",
            "   • Develop corrective action plans for Medium Risk studies",
            "   • Assign dedicated CRA resources to critical sites",
            "   • Schedule enhanced monitoring visits",
            "   • Update stakeholders on remediation progress",
            "",
            "3. ONGOING:",
            "   • Track DQI improvements weekly",
            "   • Re-run this analysis every 7-14 days to monitor trends",
            "   • Document all corrective actions in study binder",
            "   • Share risk trends with DMC and regulatory teams",
            "",
            "NOTE: This analysis is data-driven, explainable, and automated.",
            "All DQI scores are transparent (based on explicit signal weights).",
            "Regular re-runs enable early detection of emerging issues.",
        ]
        
        for rec in recommendations:
            f.write(rec + "\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Executive summary saved to: {output_path}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STEP 5: EXECUTIVE SUMMARY & RECOMMENDATIONS GENERATION")
    print("=" * 80)
    
    # Load risk rankings
    study_ranks_path = "outputs/risk_rankings.csv"
    site_ranks_path = "outputs/risk_rankings_site_level.csv"
    
    if not os.path.exists(study_ranks_path) or not os.path.exists(site_ranks_path):
        print(f"\n❌ Risk ranking files not found. Run risk_ranking.py first.")
    else:
        study_ranks = pd.read_csv(study_ranks_path)
        site_ranks = pd.read_csv(site_ranks_path)
        
        print(f"\nLoaded risk rankings for {len(study_ranks)} studies and {len(site_ranks)} sites")
        
        # Generate executive summary
        print("\nGenerating executive summary...")
        generate_executive_summary(study_ranks, site_ranks)
        
        # Print to console as well
        with open("outputs/executive_summary.txt", 'r') as f:
            print("\n" + f.read())
        
        print("\n✓ Summary generation complete.\n")