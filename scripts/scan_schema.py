"""
STEP 1: SCHEMA DISCOVERY & EXTRACTION (UNCHANGED)
=================================================

Status: NO CHANGES FROM ORIGINAL
This module is unaffected by ML integration.
It simply discovers available files and extracts column names.
"""

import os
import pandas as pd
from pathlib import Path
import json

def scan_schema(data_dir="data"):
    """
    Recursively discover all CSV/XLS/XLSX/XLSM files and extract column names.
    
    Args:
        data_dir (str): Root directory containing study folders
    
    Returns:
        dict: {file_path: [list of column names], ...}
    """
    
    schema_map = {}
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ ERROR: Directory '{data_dir}' not found.")
        return schema_map
    
    # Find all files recursively
    excel_extensions = {'.xls', '.xlsx', '.xlsm'}
    csv_extensions = {'.csv'}
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_lower = file.lower()
            
            is_excel = any(file_lower.endswith(ext) for ext in excel_extensions)
            is_csv = any(file_lower.endswith(ext) for ext in csv_extensions)
            
            if not (is_excel or is_csv):
                continue
            
            full_path = Path(root) / file
            rel_path = full_path.relative_to(data_path)
            
            try:
                if is_csv:
                    df = pd.read_csv(full_path, nrows=0, encoding='utf-8')
                    columns = df.columns.tolist()
                    
                elif is_excel:
                    xls = pd.ExcelFile(full_path)
                    first_sheet = xls.sheet_names[0] if xls.sheet_names else None
                    
                    if first_sheet:
                        df = pd.read_excel(full_path, sheet_name=first_sheet, nrows=0)
                        columns = df.columns.tolist()
                    else:
                        columns = []
                
                if columns:
                    schema_map[str(rel_path)] = columns
                    print(f"✓ {rel_path}: {len(columns)} columns found")
                else:
                    print(f"⚠ {rel_path}: No columns detected (empty or unreadable)")
            
            except Exception as e:
                print(f"⚠ {rel_path}: Could not read ({type(e).__name__})")
                continue
    
    return schema_map


def save_schema_summary(schema_map, output_path="outputs/schema_summary.txt"):
    """
    Format and save schema map to human-readable text file.
    
    Args:
        schema_map (dict): {file_path: [columns], ...}
        output_path (str): Where to save summary
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CLINICAL TRIAL DATA SCHEMA DISCOVERY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total files discovered: {len(schema_map)}\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        
        # Group by study folder
        studies = {}
        for file_path in sorted(schema_map.keys()):
            parts = file_path.split(os.sep)
            study_folder = next((p for p in parts if p.startswith("Study_")), "Unknown")
            
            if study_folder not in studies:
                studies[study_folder] = []
            studies[study_folder].append(file_path)
        
        # Write organized by study
        for study in sorted(studies.keys()):
            f.write(f"\n{'='*80}\n")
            f.write(f"STUDY: {study}\n")
            f.write(f"{'='*80}\n")
            
            for file_path in sorted(studies[study]):
                columns = schema_map[file_path]
                f.write(f"\nFile: {file_path}\n")
                f.write(f"  Columns ({len(columns)}):\n")
                
                for col in columns:
                    f.write(f"    - {col}\n")
    
    print(f"\n✓ Schema summary saved to: {output_path}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STEP 1: SCHEMA DISCOVERY")
    print("=" * 80)
    
    schema_map = scan_schema(data_dir="data")
    
    if not schema_map:
        print("\n❌ No files discovered. Check folder structure and file types.")
    else:
        print(f"\n✓ Successfully discovered {len(schema_map)} files")
        
        save_schema_summary(schema_map, output_path="outputs/schema_summary.txt")
        
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/schema_map.json", 'w', encoding='utf-8') as f:
            json.dump(schema_map, f, indent=2)
        
        print("✓ Schema map (JSON) saved for downstream processing\n")