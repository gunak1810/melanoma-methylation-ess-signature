import os
import pandas as pd
import numpy as np
import logging
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxnetSurvivalAnalysis
from utils.helpers import load_config, init_paths, setup_logging

def main():
    logger = setup_logging("04_feature_selection")
    cfg = load_config()
    paths = init_paths()
    rds_dir = paths["rds"]
    
    logger.info("Loading data for feature selection...")
    expr_df = pd.read_parquet(os.path.join(rds_dir, "expr_matrix.parquet"))
    clin_df = pd.read_parquet(os.path.join(rds_dir, "tcga_clinical.parquet"))
    reg_genes_path = os.path.join(rds_dir, "meth_expr_regulatory_genes.csv")
    
    if os.path.exists(reg_genes_path):
        reg_genes = pd.read_csv(reg_genes_path).iloc[:, 0].tolist()
    else:
        logger.error("Regulatory genes not found.")
        return
        
    from utils.helpers import get_tcga_patient_id
    clin_df.index = [get_tcga_patient_id(idx) for idx in clin_df.index]
    clin_df = clin_df[~clin_df.index.duplicated(keep="first")]
    
    common_patients = expr_df.columns.intersection(clin_df.index)
    expr_df = expr_df[common_patients]
    clin_df = clin_df.loc[common_patients]
    
    valid_surv = clin_df["OS_time"].notna() & clin_df["OS_event"].notna() & (clin_df["OS_time"] > 0)
    expr_df_full = expr_df.loc[:, valid_surv]
    clin_df_full = clin_df.loc[valid_surv]
    
    # Load Train Samples
    train_samples_path = os.path.join(rds_dir, "train_samples.csv")
    if os.path.exists(train_samples_path):
        train_samples = pd.read_csv(train_samples_path)["Sample"].tolist()
        # Convert train_samples to match clin_df_full index format if necessary
        # Assuming train_samples from 03 are the same format as expr_df.columns
        train_patients = [p for p in train_samples if p in clin_df_full.index]
        expr_df = expr_df_full[train_patients]
        clin_df = clin_df_full.loc[train_patients]
    else:
        logger.error("No train_samples.csv found. Run 03 first.")
        return
    
    logger.info(f"Training set patients with survival data: {clin_df.shape[0]} / {clin_df_full.shape[0]} total")
    
    candidate_genes = [g for g in reg_genes if g in expr_df.index]
    logger.info(f"Candidate regulatory genes: {len(candidate_genes)}")
    
    logger.info("1. Applying variance filter...")
    var_percentile = cfg["feature_selection"]["variance_percentile"]
    gene_var = expr_df.loc[candidate_genes].var(axis=1)
    threshold = np.percentile(gene_var, (1 - var_percentile) * 100)
    var_genes = gene_var[gene_var >= threshold].index.tolist()
    logger.info(f"Genes passing variance filter: {len(var_genes)}")
    
    logger.info("2. Applying Univariate Cox screening...")
    cox_results = []
    surv_data = clin_df[["OS_time", "OS_event"]].copy()
    
    for i, gene in enumerate(var_genes):
        if i % 500 == 0 and i > 0:
            logger.info(f"  Processed {i}/{len(var_genes)} genes...")
            
        surv_data["expr"] = expr_df.loc[gene].values
        cph = CoxPHFitter()
        try:
            cph.fit(surv_data, duration_col="OS_time", event_col="OS_event")
            p_val = cph.summary.loc["expr", "p"]
            cox_results.append({"gene": gene, "pvalue": p_val})
        except Exception:
            pass
            
    res_df = pd.DataFrame(cox_results)
    if res_df.empty:
        logger.error("No genes passed Univariate Cox screening. Halting.")
        return
        
    p_val_thresh = 0.05
    sig_genes = res_df[res_df["pvalue"] < p_val_thresh].sort_values("pvalue")
    
    if sig_genes.shape[0] > 200:
        logger.info(f"Taking top 200 genes out of {sig_genes.shape[0]} significant genes to avoid LASSO hang.")
        sig_genes = sig_genes.head(200)
        
    cox_genes = sig_genes["gene"].tolist()
    logger.info(f"Genes passing Univariate Cox (top {len(cox_genes)}): {len(cox_genes)}")
    
    if len(cox_genes) == 0:
        logger.error("No genes passed Univariate Cox screening. Halting.")
        return
        
    logger.info("3. Running LASSO Cox Regression (3-fold CV)...")
    
    X = expr_df.loc[cox_genes].T
    
    y = np.array([(bool(e), t) for e, t in zip(clin_df["OS_event"], clin_df["OS_time"])],
                 dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
                 
    coxnet = CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True, max_iter=10000)
    
    from sklearn.model_selection import GridSearchCV
    alphas = 10. ** np.linspace(-4, 1, 30) # fewer alphas
    cv = GridSearchCV(coxnet, {"alphas": [[a] for a in alphas]}, cv=3, error_score=0.5, n_jobs=-1)
    
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        cv.fit(X, y)
        
    best_model = cv.best_estimator_
    best_alpha = cv.best_params_["alphas"][0]
    
    logger.info(f"Best alpha selected: {best_alpha}")
    
    coefs = pd.Series(best_model.coef_[:, 0], index=X.columns)
    ess_genes = coefs[coefs != 0]
    
    logger.info(f"Selected ESS Genes (non-zero coefficients): {len(ess_genes)}")
    for gene, beta in ess_genes.items():
        logger.info(f"  {gene}: {beta:.4f}")
        
    logger.info("4. Computing Epigenetic State Score (ESS) for FULL cohort...")
    X_full = expr_df_full.loc[ess_genes.index].T
    ess_scores = X_full.dot(ess_genes.values)
    
    logger.info("--- MODEL LOCK ---")
    logger.info("Genes and Coefficients frozen.")
    
    ess_genes.to_csv(os.path.join(rds_dir, "ess_coefficients.csv"))
    pd.Series(ess_scores, index=X_full.index).to_csv(os.path.join(rds_dir, "ess_scores.csv"))
    
    clin_df_full["ESS"] = ess_scores.values
    clin_df_full["ESS_group"] = np.where(clin_df_full["ESS"] > clin_df_full["ESS"].median(), "High", "Low")
    clin_df_full.to_parquet(os.path.join(rds_dir, "tcga_clinical_with_ess.parquet"))
    
    logger.info("Feature Selection complete.")

if __name__ == "__main__":
    main()
