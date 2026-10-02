import os
import pandas as pd
import numpy as np
import logging
from scipy import stats
from utils.helpers import load_config, init_paths, setup_logging

def main():
    logger = setup_logging("13_depmap")
    cfg = load_config()
    paths = init_paths()
    rds_dir = paths["rds"]
    
    logger.info("Loading ESS genes and coefficients...")
    ess_genes_path = os.path.join(rds_dir, "ess_coefficients.csv")
    if not os.path.exists(ess_genes_path):
        logger.error("Model not locked. Run 04_feature_selection.py first.")
        return
        
    ess_coefs = pd.read_csv(ess_genes_path, index_col=0).iloc[:, 0]
    ess_genes = ess_coefs.index.tolist()
    
    crispr_path = os.path.join(paths["data"], "..", "dep map", "26 q1 CRISPRGeneEffect.csv")
    model_path = os.path.join(paths["data"], "..", "dep map", "26Q1 Model.csv")
    
    if not os.path.exists(crispr_path) or not os.path.exists(model_path):
        logger.error("Local DepMap files not found.")
        return
        
    logger.info("Loading DepMap Model Info...")
    samples = pd.read_csv(model_path)
    
    # Filter for Melanoma (Skin lineage)
    melanoma_lines = samples[samples["OncotreeLineage"] == "Skin"]["ModelID"].tolist()
    logger.info(f"Found {len(melanoma_lines)} Melanoma cell lines.")
    
    logger.info("Loading CRISPR Chronos scores...")
    crispr = pd.read_csv(crispr_path, index_col=0)
    
    # Filter to melanoma only
    crispr_mel = crispr[crispr.index.isin(melanoma_lines)]
    
    # Map column names "GENE (ID)" to "GENE"
    crispr_mel.columns = [c.split(" ")[0] for c in crispr_mel.columns]
    
    # === ANALYSIS 1: Per-gene dependency statistics ===
    logger.info("=" * 50)
    logger.info("ANALYSIS 1: Per-Gene Dependency in Melanoma")
    logger.info("=" * 50)
    
    threshold = cfg["depmap"].get("dependency_threshold", -0.5)
    results = []
    
    for gene in ess_genes:
        if gene in crispr_mel.columns:
            scores = crispr_mel[gene].dropna()
            mean_chronos = scores.mean()
            median_chronos = scores.median()
            frac = (scores <= threshold).sum() / len(scores) if len(scores) > 0 else 0
            min_chronos = scores.min()
            
            results.append({
                "gene": gene,
                "lasso_beta": ess_coefs[gene],
                "mean_chronos": mean_chronos,
                "median_chronos": median_chronos,
                "min_chronos": min_chronos,
                "frac_dependent": frac,
                "n_lines": len(scores)
            })
        else:
            logger.warning(f"{gene} not found in DepMap.")
            
    depmap_df = pd.DataFrame(results).sort_values("mean_chronos")
    
    essential_genes = depmap_df[depmap_df["mean_chronos"] <= threshold]
    logger.info(f"Highly dependent ESS genes (mean Chronos <= {threshold}): {essential_genes.shape[0]}")
    
    # Report all genes with their dependency
    for _, row in depmap_df.iterrows():
        tag = "ESSENTIAL" if row["mean_chronos"] <= threshold else ("partial" if row["frac_dependent"] > 0.1 else "")
        logger.info(f"  {row['gene']:20s}  beta={row['lasso_beta']:+.4f}  Chronos={row['mean_chronos']:+.3f}  frac_dep={row['frac_dependent']:.2f}  {tag}")
    
    depmap_df.to_csv(os.path.join(rds_dir, "depmap_dependency_summary.csv"), index=False)
    
    # === ANALYSIS 2: Protective vs Risk gene dependency comparison ===
    logger.info("=" * 50)
    logger.info("ANALYSIS 2: Protective vs Risk Gene Dependency")
    logger.info("=" * 50)
    
    protective_genes = [g for g in ess_genes if ess_coefs[g] < 0 and g in crispr_mel.columns]
    risk_genes = [g for g in ess_genes if ess_coefs[g] > 0 and g in crispr_mel.columns]
    
    if protective_genes and risk_genes:
        prot_chronos = crispr_mel[protective_genes].mean(axis=1).dropna()
        risk_chronos = crispr_mel[risk_genes].mean(axis=1).dropna()
        
        logger.info(f"Protective genes (n={len(protective_genes)}): mean dependency = {prot_chronos.mean():.3f}")
        logger.info(f"Risk genes (n={len(risk_genes)}): mean dependency = {risk_chronos.mean():.3f}")
        
        stat, pval = stats.mannwhitneyu(prot_chronos, risk_chronos, alternative='two-sided')
        logger.info(f"Mann-Whitney U (protective vs risk dependency): p = {pval:.4f}")
        
        if risk_chronos.mean() < prot_chronos.mean():
            logger.info("--> Risk genes show STRONGER dependency (more essential for melanoma viability)")
        else:
            logger.info("--> Protective (immune) genes show STRONGER dependency")
    
    # === ANALYSIS 3: Lineage-specific essentiality (melanoma vs pan-cancer) ===
    logger.info("=" * 50)
    logger.info("ANALYSIS 3: Melanoma-Specific vs Pan-Cancer Essentiality")
    logger.info("=" * 50)
    
    crispr_all = crispr.copy()
    crispr_all.columns = [c.split(" ")[0] for c in crispr_all.columns]
    non_mel = crispr_all[~crispr_all.index.isin(melanoma_lines)]
    
    lineage_results = []
    for gene in ess_genes:
        if gene in crispr_mel.columns and gene in non_mel.columns:
            mel_scores = crispr_mel[gene].dropna()
            other_scores = non_mel[gene].dropna()
            
            if len(mel_scores) > 5 and len(other_scores) > 5:
                stat, pval = stats.mannwhitneyu(mel_scores, other_scores, alternative='two-sided')
                diff = mel_scores.mean() - other_scores.mean()
                lineage_results.append({
                    "gene": gene,
                    "mel_chronos": mel_scores.mean(),
                    "other_chronos": other_scores.mean(),
                    "diff": diff,
                    "pvalue": pval,
                    "direction": "Melanoma-specific" if diff < -0.1 else ("Pan-essential" if diff > 0.1 else "Similar")
                })
    
    lineage_df = pd.DataFrame(lineage_results).sort_values("diff")
    
    for _, row in lineage_df.iterrows():
        sig = "***" if row["pvalue"] < 0.001 else "**" if row["pvalue"] < 0.01 else "*" if row["pvalue"] < 0.05 else "ns"
        logger.info(f"  {row['gene']:20s}  Mel={row['mel_chronos']:+.3f}  Other={row['other_chronos']:+.3f}  diff={row['diff']:+.3f}  p={row['pvalue']:.4f} {sig}  [{row['direction']}]")
    
    lineage_df.to_csv(os.path.join(rds_dir, "depmap_lineage_specificity.csv"), index=False)
    
    logger.info("DepMap dependency analysis complete.")

if __name__ == "__main__":
    main()
