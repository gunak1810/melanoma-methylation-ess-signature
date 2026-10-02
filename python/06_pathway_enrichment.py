import os
import pandas as pd
import numpy as np
from scipy import stats
import gseapy as gp
import logging
from statsmodels.stats.multitest import multipletests
from utils.helpers import load_config, init_paths, setup_logging
from utils.gene_sets import get_msigdb_sets

def run_ssgsea(expr_df, gene_sets, outdir, name):
    """Run ssGSEA using gseapy."""
    # gseapy expects genes in rows, samples in columns
    ss = gp.ssgsea(data=expr_df, gene_sets=gene_sets, outdir=os.path.join(outdir, name),
                   sample_norm_method='rank', min_size=15, max_size=500, permutation_num=0,
                   no_plot=True, processes=4, format='png')
    
    # Extract results
    return ss.res2d

def kw_test_across_states(df_scores, states):
    """Kruskal-Wallis test across groups."""
    results = []
    
    unique_states = states.unique()
    
    for feature in df_scores.index:
        groups = [df_scores.loc[feature, states == s].dropna().values for s in unique_states]
        # Only test if all groups have data
        if all(len(g) > 2 for g in groups):
            h_stat, p_val = stats.kruskal(*groups)
            results.append({
                "feature": feature,
                "kw_statistic": h_stat,
                "pvalue": p_val
            })
            
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        _, fdr, _, _ = multipletests(res_df["pvalue"], method="fdr_bh")
        res_df["fdr"] = fdr
        res_df = res_df.sort_values("pvalue")
        
    return res_df

def main():
    logger = setup_logging("06_pathway_enrichment")
    cfg = load_config()
    paths = init_paths()
    rds_dir = paths["rds"]
    
    logger.info("Loading data...")
    expr_df = pd.read_parquet(os.path.join(rds_dir, "expr_matrix.parquet"))
    states_series = pd.read_csv(os.path.join(rds_dir, "patient_states.csv"), index_col=0).iloc[:, 0]
    
    # Align
    common = expr_df.columns.intersection(states_series.index)
    expr_df = expr_df[common]
    states = states_series.loc[common]
    
    # Load gene sets
    logger.info("Fetching MSigDB gene sets...")
    hm_sets = get_msigdb_sets("H")
    
    if not hm_sets:
        logger.error("Could not load Hallmark gene sets. Ensure internet connection or update gseapy.")
        return
        
    # 1. ssGSEA Enrichment
    logger.info("Running ssGSEA for Hallmark pathways...")
    # ssGSEA returns samples x pathways, we transpose to pathways x samples
    ssgsea_res = run_ssgsea(expr_df, hm_sets, rds_dir, "ssgsea_hallmark")
    
    ssgsea_scores = ssgsea_res.pivot(index='Term', columns='Name', values='NES')
    ssgsea_scores.to_parquet(os.path.join(rds_dir, "ssgsea_hallmark.parquet"))
    
    # 2. State Comparisons
    logger.info("Comparing pathway enrichment across states...")
    kw_results = kw_test_across_states(ssgsea_scores, states)
    
    kw_results.to_csv(os.path.join(rds_dir, "pathway_state_tests.csv"), index=False)
    
    logger.info(f"Significant pathways (FDR < 0.05): {(kw_results['fdr'] < 0.05).sum()}")
    
    # Print top
    for _, row in kw_results.head(10).iterrows():
        logger.info(f"  {row['feature']}: p={row['pvalue']:.2e}, FDR={row['fdr']:.2e}")
        
    # 3. Prerank GSEA (State specific)
    logger.info("Running state-specific Prerank GSEA...")
    gsea_results = []
    
    for state in states.unique():
        logger.info(f"  GSEA for {state} vs Rest...")
        
        # Calculate t-statistic equivalent (Welch's t-test)
        is_state = states == state
        group1 = expr_df.loc[:, is_state]
        group2 = expr_df.loc[:, ~is_state]
        
        t_stat, p_val = stats.ttest_ind(group1, group2, axis=1, equal_var=False)
        
        # Rank by t-statistic
        ranks = pd.Series(t_stat, index=expr_df.index).dropna().sort_values(ascending=False)
        
        # Run prerank
        pre_res = gp.prerank(rnk=ranks, gene_sets=hm_sets,
                             threads=4, min_size=15, max_size=500, permutation_num=1000,
                             outdir=None, seed=cfg["project"]["seed"], no_plot=True)
                             
        res2d = pre_res.res2d
        res2d["state"] = state
        gsea_results.append(res2d)
        
    gsea_df = pd.concat(gsea_results)
    # Drop list columns for saving
    if 'Lead_genes' in gsea_df.columns:
        gsea_df = gsea_df.drop(columns=['Lead_genes'])
        
    gsea_df.to_parquet(os.path.join(rds_dir, "gsea_state_specific.parquet"))
    
    logger.info("Pathway Enrichment complete.")

if __name__ == "__main__":
    main()
