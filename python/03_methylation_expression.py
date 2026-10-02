import os
import pandas as pd
import numpy as np
from scipy import stats
import logging
import urllib.request
from statsmodels.stats.multitest import multipletests
from utils.helpers import load_config, init_paths, setup_logging

def main():
    logger = setup_logging("03_methylation_expression")
    cfg = load_config()
    paths = init_paths()
    rds_dir = paths["rds"]
    
    logger.info("Loading preprocessed matrices...")
    expr_df = pd.read_parquet(os.path.join(rds_dir, "expr_matrix.parquet"))
    meth_df = pd.read_parquet(os.path.join(rds_dir, "meth_matrix.parquet"))
    
    ann_path = os.path.join(rds_dir, "meth_annotation.parquet")
    if not os.path.exists(ann_path):
        logger.info("Downloading Illumina 450K manifest...")
        manifest_url = "https://webdata.illumina.com/downloads/productfiles/humanmethylation450/humanmethylation450_15017482_v1-2.csv"
        csv_path = os.path.join(paths["data"], "humanmethylation450.csv")
        if not os.path.exists(csv_path):
            urllib.request.urlretrieve(manifest_url, csv_path)
            
        logger.info("Parsing manifest...")
        # Read the manifest, skipping the first 7 lines
        ann_df = pd.read_csv(csv_path, skiprows=7, usecols=["IlmnID", "UCSC_RefGene_Name", "UCSC_RefGene_Group"], dtype=str)
        # Drop rows where gene is missing
        ann_df = ann_df.dropna(subset=["UCSC_RefGene_Name"])
        
        # A CpG can map to multiple genes (semicolon separated). Explode them.
        ann_df["UCSC_RefGene_Name"] = ann_df["UCSC_RefGene_Name"].str.split(";")
        ann_df["UCSC_RefGene_Group"] = ann_df["UCSC_RefGene_Group"].str.split(";")
        
        ann_exploded = ann_df.explode(["UCSC_RefGene_Name", "UCSC_RefGene_Group"])
        # Keep unique CpG-Gene pairs
        ann_exploded = ann_exploded.drop_duplicates(subset=["IlmnID", "UCSC_RefGene_Name"])
        
        meth_ann = pd.DataFrame({
            "cpg": ann_exploded["IlmnID"],
            "gene": ann_exploded["UCSC_RefGene_Name"],
            "genomic_context": ann_exploded["UCSC_RefGene_Group"]
        })
        meth_ann.to_parquet(ann_path)
    else:
        meth_ann = pd.read_parquet(ann_path)
        
    logger.info(f"Loaded {meth_ann.shape[0]} valid CpG-Gene annotation pairs.")
    
    # Align samples
    common_samples = expr_df.columns.intersection(meth_df.columns)
    
    # --- Train/Test Split (Prevent Data Leakage) ---
    from sklearn.model_selection import train_test_split
    logger.info("Splitting cohort into 70% Train and 30% Test...")
    train_samples, test_samples = train_test_split(list(common_samples), test_size=0.3, random_state=cfg.get("project", {}).get("seed", 42))
    
    pd.Series(train_samples).to_csv(os.path.join(rds_dir, "train_samples.csv"), index=False, header=["Sample"])
    pd.Series(test_samples).to_csv(os.path.join(rds_dir, "test_samples.csv"), index=False, header=["Sample"])
    
    # Restrict to Training Set for Epigenetic Filtering
    expr_df_train = expr_df[train_samples]
    meth_df_train = meth_df[train_samples]
    
    logger.info(f"Testing combinations across {len(train_samples)} training samples...")
    results = []
    
    # Only test pairs where both CpG and Gene are in our matrices
    valid_ann = meth_ann[meth_ann["cpg"].isin(meth_df_train.index) & meth_ann["gene"].isin(expr_df_train.index)]
    logger.info(f"Testing {valid_ann.shape[0]} filtered CpG-Gene pairs...")
    
    count = 0
    total = valid_ann.shape[0]
    for _, row in valid_ann.iterrows():
        cpg = row["cpg"]
        gene = row["gene"]
        context = row["genomic_context"]
        
        meth_vals = meth_df_train.loc[cpg].values
        expr_vals = expr_df_train.loc[gene].values
        
        valid = ~np.isnan(meth_vals) & ~np.isnan(expr_vals)
        if valid.sum() < 10:
            continue
            
        rho, pval = stats.spearmanr(meth_vals[valid], expr_vals[valid])
        
        results.append({
            "cpg": cpg,
            "gene": gene,
            "genomic_context": context,
            "rho": rho,
            "pvalue": pval
        })
        
        count += 1
        if count % 50000 == 0:
            logger.info(f"Processed {count} / {total} pairs...")
            
    res_df = pd.DataFrame(results)
    
    # FDR Correction
    logger.info("Applying FDR correction...")
    res_df = res_df.dropna(subset=["pvalue"])
    _, fdr, _, _ = multipletests(res_df["pvalue"], method="fdr_bh")
    res_df["fdr"] = fdr
    res_df["abs_rho"] = res_df["rho"].abs()
    
    res_df.to_parquet(os.path.join(rds_dir, "meth_expr_pairs_all.parquet"))
    
    # Filter significant
    min_rho = cfg["meth_expr"].get("min_abs_rho", 0.3)
    fdr_thresh = cfg["meth_expr"].get("fdr_threshold", 0.05)
    
    sig_pairs = res_df[(res_df["abs_rho"] >= min_rho) & (res_df["fdr"] < fdr_thresh)]
    logger.info(f"Significant pairs: {sig_pairs.shape[0]} / {res_df.shape[0]}")
    
    sig_pairs.to_parquet(os.path.join(rds_dir, "meth_expr_pairs_sig.parquet"))
    
    # Extract unique regulatory genes
    reg_genes = sig_pairs["gene"].unique()
    logger.info(f"Unique regulatory genes: {len(reg_genes)}")
    pd.Series(reg_genes).to_csv(os.path.join(rds_dir, "meth_expr_regulatory_genes.csv"), index=False)
    
    logger.info("Methylation-Expression Integration complete.")

if __name__ == "__main__":
    main()
