import os
import pandas as pd
import numpy as np
import logging
from sklearn.impute import KNNImputer
from utils.helpers import load_config, init_paths, setup_logging, is_tumor_sample, get_tcga_patient_id

def main():
    logger = setup_logging("02_qc_preprocessing")
    cfg = load_config()
    paths = init_paths()
    rds_dir = paths["rds"]
    
    # 1. Load Data
    logger.info("Loading RNA and Methylation matrices...")
    rna_df = pd.read_parquet(os.path.join(rds_dir, "tcga_rna.parquet"))
    meth_df = pd.read_parquet(os.path.join(rds_dir, "tcga_meth.parquet"))
    
    # TCGA Xena datasets often have samples as columns
    # We want genes/CpGs as rows, samples as columns
    if rna_df.shape[0] < rna_df.shape[1] and rna_df.index.name == 'sample':
        rna_df = rna_df.T
    if meth_df.shape[0] < meth_df.shape[1] and meth_df.index.name == 'sample':
        meth_df = meth_df.T
        
    logger.info(f"RNA raw: {rna_df.shape[0]} genes, {rna_df.shape[1]} samples")
    logger.info(f"Meth raw: {meth_df.shape[0]} CpGs, {meth_df.shape[1]} samples")
    
    # 2. Filter Tumor Samples
    logger.info("Filtering for tumor samples...")
    rna_tumor_cols = [c for c in rna_df.columns if is_tumor_sample(c)]
    meth_tumor_cols = [c for c in meth_df.columns if is_tumor_sample(c)]
    
    rna_df = rna_df[rna_tumor_cols]
    meth_df = meth_df[meth_tumor_cols]
    
    # 3. Match Patients
    logger.info("Matching patients between RNA and Methylation...")
    # Map to patient ID (first 12 chars)
    rna_patients = {c: get_tcga_patient_id(c) for c in rna_df.columns}
    meth_patients = {c: get_tcga_patient_id(c) for c in meth_df.columns}
    
    common_patients = set(rna_patients.values()).intersection(set(meth_patients.values()))
    logger.info(f"Common patients: {len(common_patients)}")
    
    # Keep one sample per patient
    def filter_one_per_patient(df, patient_map, valid_patients):
        kept_cols = []
        seen = set()
        for col, pid in patient_map.items():
            if pid in valid_patients and pid not in seen:
                kept_cols.append(col)
                seen.add(pid)
        
        filtered_df = df[kept_cols].copy()
        filtered_df.columns = [patient_map[c] for c in kept_cols]
        return filtered_df
        
    rna_matched = filter_one_per_patient(rna_df, rna_patients, common_patients)
    meth_matched = filter_one_per_patient(meth_df, meth_patients, common_patients)
    
    # Align columns
    meth_matched = meth_matched[rna_matched.columns]
    
    # 4. RNA QC
    logger.info("Running RNA-seq QC...")
    # UCSC Xena RNA is typically log2(count+1) or log2(RSEM+1)
    # We will filter low expression: average log2(x+1) > 0.5 (approx count > 0.4)
    # This is a heuristic for pre-processed Xena data
    mean_expr = rna_matched.mean(axis=1)
    rna_filtered = rna_matched[mean_expr > 0.5]
    logger.info(f"RNA after expression filter: {rna_filtered.shape[0]} genes")
    
    # 5. Methylation QC
    logger.info("Running Methylation QC...")
    max_na_frac = cfg["qc"]["methylation"].get("max_missing_fraction", 0.2)
    na_frac = meth_matched.isna().mean(axis=1)
    meth_filtered = meth_matched[na_frac <= max_na_frac]
    logger.info(f"Meth after NA filter (<{max_na_frac*100}% NA): {meth_filtered.shape[0]} CpGs")
    
    # Impute remaining missing values
    n_missing = meth_filtered.isna().sum().sum()
    if n_missing > 0:
        logger.info(f"Imputing {n_missing} missing values with KNN...")
        k = cfg["qc"]["methylation"].get("imputation_k", 10)
        imputer = KNNImputer(n_neighbors=k)
        # Imputer works on columns, we want to impute samples (so we impute over genes)
        meth_imputed = imputer.fit_transform(meth_filtered.T).T
        meth_filtered = pd.DataFrame(meth_imputed, index=meth_filtered.index, columns=meth_filtered.columns)
        
    # Save
    logger.info("Saving processed matrices...")
    rna_filtered.to_parquet(os.path.join(rds_dir, "expr_matrix.parquet"))
    meth_filtered.to_parquet(os.path.join(rds_dir, "meth_matrix.parquet"))
    pd.Series(list(common_patients)).to_csv(os.path.join(rds_dir, "matched_patients.csv"), index=False)
    
    logger.info("QC & Preprocessing complete.")

if __name__ == "__main__":
    main()
