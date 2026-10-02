import os
import pandas as pd
import numpy as np
from scipy import stats
import logging
from utils.helpers import load_config, init_paths, setup_logging
from utils.gene_sets import get_stemness_markers, get_melanoma_phenotype_genes
from statsmodels.stats.multitest import multipletests

def compute_signature_score(expr_df, pos_genes, neg_genes):
    """
    Compute a simple signature score: 
    mean(pos_genes) - mean(neg_genes)
    """
    pos_avail = [g for g in pos_genes if g in expr_df.index]
    neg_avail = [g for g in neg_genes if g in expr_df.index]
    
    if len(pos_avail) == 0 and len(neg_avail) == 0:
        return pd.Series(np.nan, index=expr_df.columns)
        
    pos_score = expr_df.loc[pos_avail].mean(axis=0) if len(pos_avail) > 0 else 0
    neg_score = expr_df.loc[neg_avail].mean(axis=0) if len(neg_avail) > 0 else 0
    
    return pos_score - neg_score

def main():
    logger = setup_logging("08_stemness")
    cfg = load_config()
    paths = init_paths()
    rds_dir = paths["rds"]
    
    logger.info("Loading data...")
    expr_df = pd.read_parquet(os.path.join(rds_dir, "expr_matrix.parquet"))
    states_series = pd.read_csv(os.path.join(rds_dir, "patient_states.csv"), index_col=0).iloc[:, 0]
    ess_scores = pd.read_csv(os.path.join(rds_dir, "ess_scores.csv"), index_col=0).iloc[:, 0]
    
    common = expr_df.columns.intersection(states_series.index)
    expr_df = expr_df[common]
    states = states_series.loc[common]
    ess = ess_scores.loc[common]
    
    # 1. Stemness Score (mRNAsi proxy)
    logger.info("Computing Stemness Scores...")
    stem_genes = get_stemness_markers()
    stemness_scores = compute_signature_score(expr_df, stem_genes["positive"], stem_genes["negative"])
    
    # Scale to 0-1
    stem_min = stemness_scores.min()
    stem_max = stemness_scores.max()
    stemness_scaled = (stemness_scores - stem_min) / (stem_max - stem_min)
    
    # 2. Melanoma Phenotype Scores
    logger.info("Computing Melanoma Phenotype Scores (e.g. AXL/MITF)...")
    pheno_genes = get_melanoma_phenotype_genes()
    
    melanocytic_score = expr_df.loc[[g for g in pheno_genes["melanocytic"] if g in expr_df.index]].mean(axis=0)
    invasive_score = expr_df.loc[[g for g in pheno_genes["invasive"] if g in expr_df.index]].mean(axis=0)
    
    # 3. Combine
    res_df = pd.DataFrame({
        "patient": common,
        "state": states.values,
        "ESS": ess.values,
        "stemness_marker": stemness_scaled.values,
        "melanocytic_score": melanocytic_score.values,
        "invasive_score": invasive_score.values,
        "AXL_MITF_ratio": (invasive_score - melanocytic_score).values # Log scale ratio
    })
    
    res_df.to_csv(os.path.join(rds_dir, "stemness_scores.csv"), index=False)
    
    # 4. Correlate with ESS
    logger.info("Correlations with ESS:")
    for metric in ["stemness_marker", "melanocytic_score", "invasive_score", "AXL_MITF_ratio"]:
        valid = res_df[metric].notna() & res_df["ESS"].notna()
        rho, pval = stats.spearmanr(res_df.loc[valid, "ESS"], res_df.loc[valid, metric])
        logger.info(f"  {metric}: rho={rho:.3f}, p={pval:.2e}")
        
    logger.info("Stemness Analysis complete.")

if __name__ == "__main__":
    main()
