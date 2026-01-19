# src/data_loader.py
# Load and merge clinical trial data from multiple sources (REAL DATA COMPATIBLE)

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, List
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, COLUMN_MAPPINGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_file_by_keywords(study_dir: Path, keywords: List[str]):
    """
    Find first file in study_dir whose name contains all keywords.
    Supports CSV and Excel files.
    """
    for file in study_dir.iterdir():
        name = file.name.lower()
        if all(k.lower() in name for k in keywords):
            return file
    return None


def read_table(file: Path) -> pd.DataFrame:
    """Read CSV or Excel transparently."""
    if file.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(file)
    return pd.read_csv(file)


class DataLoader:
    """Load and merge clinical trial data from multiple sources."""
    
    def __init__(self):
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR
        self.studies = {}
    
    def load_study(self, study_dir: Path) -> pd.DataFrame:
        """Load all relevant files for a single study."""
        
        study_name = study_dir.name
        logger.info(f"Loading {study_name}...")
        
        try:
            # ---------------- PRIMARY: EDC METRICS ----------------
            edc_file = find_file_by_keywords(
                study_dir, ["edc", "metrics"]
            )
            if not edc_file:
                logger.warning(f"  ✗ No EDC metrics file found, skipping study")
                return None
            
            df = read_table(edc_file)
            logger.info(f"  ✓ Loaded EDC metrics: {len(df)} rows")
            
            # ---------------- MISSING PAGES ----------------
            pages_file = find_file_by_keywords(
                study_dir, ["missing", "pages"]
            )
            if pages_file:
                pages_df = read_table(pages_file)
                missing_pages = pages_df.groupby('Subject').size().reset_index(
                    name='Missing_Pages_Count'
                )
                df = df.merge(missing_pages, on='Subject', how='left')
                df['Missing_Pages_Count'].fillna(0, inplace=True)
                logger.info("  ✓ Merged missing pages data")
            
            # ---------------- MISSING LABS ----------------
            labs_file = find_file_by_keywords(
                study_dir, ["missing", "lab"]
            )
            if labs_file:
                labs_df = read_table(labs_file)
                missing_labs = labs_df.groupby('Subject').size().reset_index(
                    name='Missing_Labs_Count'
                )
                df = df.merge(missing_labs, on='Subject', how='left')
                df['Missing_Labs_Count'].fillna(0, inplace=True)
                logger.info("  ✓ Merged missing lab data")
            
            # ---------------- SAE DASHBOARD ----------------
            sae_file = find_file_by_keywords(
                study_dir, ["sae"]
            )
            if sae_file:
                sae_df = read_table(sae_file)
                sae_count = sae_df.groupby('Subject').size().reset_index(
                    name='SAE_Count'
                )
                df = df.merge(sae_count, on='Subject', how='left')
                df['SAE_Count'].fillna(0, inplace=True)
                logger.info("  ✓ Merged SAE data")
            
            # ---------------- EDRR ----------------
            edrr_file = find_file_by_keywords(
                study_dir, ["edrr"]
            )
            if edrr_file:
                edrr_df = read_table(edrr_file)
                edrr_count = edrr_df.groupby('Subject').size().reset_index(
                    name='EDRR_Open_Issues'
                )
                df = df.merge(edrr_count, on='Subject', how='left')
                df['EDRR_Open_Issues'].fillna(0, inplace=True)
                logger.info("  ✓ Merged EDRR data")
            
            # ---------------- MEDDRA CODING ----------------
            meddra_file = find_file_by_keywords(
                study_dir, ["meddra"]
            )
            if meddra_file:
                meddra_df = read_table(meddra_file)
                uncoded = meddra_df[
                    meddra_df['Coding Status'].str.contains("uncoded", case=False)
                ].groupby('Subject').size().reset_index(name='MedDRA_Uncoded')
                df = df.merge(uncoded, on='Subject', how='left')
                df['MedDRA_Uncoded'].fillna(0, inplace=True)
                logger.info("  ✓ Merged MedDRA coding data")
            
            # ---------------- WHODRUG CODING ----------------
            whodrug_file = find_file_by_keywords(
                study_dir, ["whodrug"]
            )
            if whodrug_file:
                whodrug_df = read_table(whodrug_file)
                uncoded = whodrug_df[
                    whodrug_df['Coding Status'].str.contains("uncoded", case=False)
                ].groupby('Subject').size().reset_index(name='WHODrug_Uncoded')
                df = df.merge(uncoded, on='Subject', how='left')
                df['WHODrug_Uncoded'].fillna(0, inplace=True)
                logger.info("  ✓ Merged WHO Drug coding data")
            
            # ---------------- INACTIVATED FORMS ----------------
            inact_file = find_file_by_keywords(
                study_dir, ["inactivated"]
            )
            if inact_file:
                inact_df = read_table(inact_file)
                inact_count = inact_df.groupby('Subject').size().reset_index(
                    name='Inactivated_Count'
                )
                df = df.merge(inact_count, on='Subject', how='left')
                df['Inactivated_Count'].fillna(0, inplace=True)
                logger.info("  ✓ Merged inactivated forms data")
            
            # ---------------- CLEANUP ----------------
            df.columns = df.columns.str.strip()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
            self.studies[study_name] = df
            logger.info(f"  ✓ Successfully loaded {study_name}")
            return df
        
        except Exception as e:
            logger.error(f"  ✗ Error loading {study_name}: {e}")
            return None
    
    def load_all_studies(self) -> Dict[str, pd.DataFrame]:
        """Load all study folders under raw data directory."""
        
        study_dirs = [d for d in self.raw_data_dir.iterdir() if d.is_dir()]
        logger.info(f"\nLoading {len(study_dirs)} studies from {self.raw_data_dir}...")
        
        for study_dir in study_dirs:
            self.load_study(study_dir)
        
        logger.info(f"\n✓ Loaded {len(self.studies)} studies successfully\n")
        return self.studies
    
    def get_all_studies(self) -> Dict[str, pd.DataFrame]:
        return self.studies
