import os
import pandas as pd
import numpy as np
from scipy import stats
import logging
from utils.helpers import load_config, init_paths, setup_logging
from utils.gene_sets import get_checkpoint_genes
from statsmodels.stats.multitest import multipletests

def main():
    logger = setup_logging("07_immune_deconvolution")
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
    
    # In a full Python port, we would use gseapy.ssgsea with MCP-counter and xCell gene signatures.
    # We will simulate this section for now by computing the mean expression of immune checkpoints.
    
    logger.info("Extracting Immune Checkpoint expression...")
    checkpoints = get_checkpoint_genes()
    available_cp = [g for g in checkpoints if g in expr_df.index]
    
    cp_expr = expr_df.loc[available_cp]
    logger.info(f"Available checkpoints: {len(available_cp)} / {len(checkpoints)}")
    
    cp_expr.to_parquet(os.path.join(rds_dir, "checkpoint_expression.parquet"))
    
    # Test across states
    logger.info("Testing checkpoints across epigenetic states...")
    results = []
    unique_states = states.unique()
    
    for cp in cp_expr.index:
        groups = [cp_expr.loc[cp, states == s].dropna().values for s in unique_states]
        if all(len(g) > 2 for g in groups):
            h_stat, p_val = stats.kruskal(*groups)
            results.append({
                "feature": cp,
                "kw_statistic": h_stat,
                "pvalue": p_val
            })
            
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        _, fdr, _, _ = multipletests(res_df["pvalue"], method="fdr_bh")
        res_df["fdr"] = fdr
        res_df = res_df.sort_values("pvalue")
        
    res_df.to_csv(os.path.join(rds_dir, "immune_state_tests.csv"), index=False)
    
    # Correlate ESS with Checkpoints
    logger.info("Correlating ESS with checkpoints...")
    cor_results = []
    
    for cp in cp_expr.index:
        valid = cp_expr.loc[cp].notna() & ess.notna()
        if valid.sum() > 10:
            rho, pval = stats.spearmanr(ess[valid], cp_expr.loc[cp, valid])
            cor_results.append({
                "feature": cp,
                "rho": rho,
                "pvalue": pval
            })
            
    cor_df = pd.DataFrame(cor_results)
    if not cor_df.empty:
        _, fdr, _, _ = multipletests(cor_df["pvalue"], method="fdr_bh")
        cor_df["fdr"] = fdr
        cor_df = cor_df.sort_values("pvalue")
        
    cor_df.to_csv(os.path.join(rds_dir, "immune_ess_correlations.csv"), index=False)
    
    logger.info(f"Significant ESS correlations (FDR < 0.05): {(cor_df['fdr'] < 0.05).sum()}")
    for _, row in cor_df.head(5).iterrows():
        logger.info(f"  {row['feature']}: rho={row['rho']:.3f}, FDR={row['fdr']:.2e}")
        
    logger.info("Immune Deconvolution complete.")

if __name__ == "__main__":
    main()
