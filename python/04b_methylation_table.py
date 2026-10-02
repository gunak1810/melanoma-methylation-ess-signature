import os
import pandas as pd
import logging
from utils.helpers import load_config, init_paths, setup_logging

def main():
    logger = setup_logging("04b_methylation_table")
    cfg = load_config()
    paths = init_paths()
    rds_dir = paths["rds"]
    
    logger.info("Generating explicit Epigenetic Evidence Table...")
    
    # Load ESS coefficients
    ess_path = os.path.join(rds_dir, "ess_coefficients.csv")
    if not os.path.exists(ess_path):
        logger.error("ess_coefficients.csv not found.")
        return
    ess_coefs = pd.read_csv(ess_path, index_col=0).iloc[:, 0]
    ess_genes = ess_coefs.index.tolist()
    
    # Load Significant Meth-Expr pairs
    pairs_path = os.path.join(rds_dir, "meth_expr_pairs_sig.parquet")
    if not os.path.exists(pairs_path):
        logger.error("meth_expr_pairs_sig.parquet not found.")
        return
    sig_pairs = pd.read_parquet(pairs_path)
    
    # Filter for ESS genes
    ess_pairs = sig_pairs[sig_pairs["gene"].isin(ess_genes)].copy()
    
    # If a gene has multiple CpGs, pick the one with the strongest absolute correlation
    ess_pairs = ess_pairs.sort_values("abs_rho", ascending=False).drop_duplicates("gene")
    
    # Add LASSO beta
    ess_pairs["lasso_beta"] = ess_pairs["gene"].map(ess_coefs)
    
    # Sort by LASSO beta for easy interpretation
    ess_pairs = ess_pairs.sort_values("lasso_beta")
    
    # Select columns for table
    table_df = ess_pairs[["gene", "cpg", "genomic_context", "rho", "fdr", "lasso_beta"]]
    table_df.columns = ["Gene", "CpG Probe", "Region", "Methylation-Expression rho", "FDR", "LASSO Beta"]
    
    out_path = os.path.join(paths["tables"], "Table_1_CpG_Probes.csv")
    os.makedirs(paths["tables"], exist_ok=True)
    table_df.to_csv(out_path, index=False)
    
    logger.info(f"Saved Epigenetic Evidence Table to {out_path}")
    logger.info("Complete.")

if __name__ == "__main__":
    main()
