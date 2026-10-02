import os
import pandas as pd
import numpy as np
import logging
import urllib.request
from scipy.stats import ranksums, mannwhitneyu
from lifelines import CoxPHFitter
from utils.helpers import load_config, init_paths, setup_logging
from utils.geo_parser import fetch_and_parse_geo

def process_gse78220(data_dir, ess_coefs, rds_dir):
    logger = logging.getLogger("11_immunotherapy")
    logger.info("Downloading GSE78220 (Hugo et al. Anti-PD1 Melanoma)...")
    
    # Download Series Matrix for Phenotype
    _, pheno = fetch_and_parse_geo("GSE78220", data_dir)
    
    # Download FPKM for Expression
    expr_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE78nnn/GSE78220/suppl/GSE78220_PatientFPKM.xlsx"
    expr_path = os.path.join(data_dir, "GSE78220_PatientFPKM.xlsx")
    if not os.path.exists(expr_path):
        logger.info("Downloading GSE78220 expression FPKM matrix...")
        urllib.request.urlretrieve(expr_url, expr_path)
        
    expr = pd.read_excel(expr_path, index_col=0)
    
    # Calculate ESS
    common = [g for g in ess_coefs.index if g in expr.index]
    if len(common) < len(ess_coefs):
        logger.warning(f"Missing {len(ess_coefs) - len(common)} ESS genes in GSE78220.")
        
    ess_scores = expr.loc[common].T.dot(ess_coefs.loc[common].values)
    
    # Parse Response AND Survival from Pheno
    # Format: "patient id: Pt1 | anti-pd-1 response: Progressive Disease | overall survival (days): 607 | vital status: Dead | ..."
    import re
    
    response_data = {}  # GSM -> {response, os_days, vital_status, patient_id}
    char_cols = [c for c in pheno.columns if 'characteristics' in c]
    
    for gsm in pheno.index:
        rec = {"response": None, "os_days": None, "vital_status": None, "patient_id": None}
        for c in char_cols:
            full_val = str(pheno.loc[gsm, c])
            for val in full_val.split('|'):
                val_clean = val.strip().lower()
                if 'patient id' in val_clean:
                    rec["patient_id"] = val.split(':')[-1].strip()
                elif 'anti-pd-1 response' in val_clean or 'response' in val_clean:
                    resp = val.split(':')[-1].strip()
                    if resp.lower() in ['progressive disease', 'pd']:
                        rec["response"] = "NR"
                    elif resp.lower() in ['partial response', 'complete response', 'pr', 'cr', 'stable disease', 'sd']:
                        rec["response"] = "R"
                elif 'overall survival' in val_clean and 'days' in val_clean:
                    try:
                        rec["os_days"] = float(val.split(':')[-1].strip())
                    except:
                        pass
                elif 'vital status' in val_clean:
                    vs = val.split(':')[-1].strip().lower()
                    if vs in ['dead', 'deceased']:
                        rec["vital_status"] = 1
                    elif vs in ['alive', 'living']:
                        rec["vital_status"] = 0
        response_data[gsm] = rec
    
    # Build mapping: patient_id (e.g. "Pt1.baseline") -> GSM ID and response
    pt_to_gsm = {}
    for gsm, rec in response_data.items():
        if rec["patient_id"]:
            # Expr matrix uses IDs like "Pt1.baseline"
            pt_to_gsm[rec["patient_id"] + ".baseline"] = gsm
            pt_to_gsm[rec["patient_id"]] = gsm
    
    logger.info(f"Mapped {len(pt_to_gsm)} patient IDs to GSM accessions")
    
    # Match ESS scores (keyed by patient ID like "Pt1") to responses (keyed by GSM)
    r_scores = []
    nr_scores = []
    r_labels = []
    
    for pat in ess_scores.index:
        gsm = pt_to_gsm.get(pat)
        if gsm and gsm in response_data:
            resp = response_data[gsm]["response"]
            if resp == "R":
                r_scores.append(ess_scores[pat])
                r_labels.append(1)
            elif resp == "NR":
                nr_scores.append(ess_scores[pat])
                r_labels.append(0)
                
    if r_scores and nr_scores:
        stat, pval = mannwhitneyu(r_scores, nr_scores, alternative='two-sided')
        logger.info(f"GSE78220 Anti-PD1 Response Analysis:")
        logger.info(f"  Responders (n={len(r_scores)}): mean ESS = {np.mean(r_scores):.3f}")
        logger.info(f"  Non-Responders (n={len(nr_scores)}): mean ESS = {np.mean(nr_scores):.3f}")
        logger.info(f"  Mann-Whitney U p-value: {pval:.4f}")
        
        if np.mean(r_scores) < np.mean(nr_scores):
            logger.info("  --> Lower ESS associated with anti-PD1 response (immune-hot)")
        else:
            logger.info("  --> Higher ESS associated with anti-PD1 response")
    else:
        logger.warning(f"Failed to match response criteria. R={len(r_scores)}, NR={len(nr_scores)}")
        
    # Also do survival analysis within the ICB cohort
    surv_rows = []
    for pat in ess_scores.index:
        gsm = pt_to_gsm.get(pat)
        if gsm and gsm in response_data:
            rec = response_data[gsm]
            if rec["os_days"] is not None and rec["vital_status"] is not None:
                surv_rows.append({
                    "patient": pat,
                    "ESS": ess_scores[pat],
                    "OS_time": rec["os_days"],
                    "OS_event": rec["vital_status"]
                })
    
    if surv_rows:
        surv_df = pd.DataFrame(surv_rows)
        logger.info(f"ICB cohort survival analysis (n={len(surv_df)})...")
        try:
            cph = CoxPHFitter()
            cph.fit(surv_df[["OS_time", "OS_event", "ESS"]], duration_col="OS_time", event_col="OS_event")
            hr = cph.summary.loc["ESS", "exp(coef)"]
            p = cph.summary.loc["ESS", "p"]
            logger.info(f"  ESS Cox HR = {hr:.2f}, p = {p:.4f}")
        except Exception as e:
            logger.error(f"  Cox failed: {e}")

def main():
    logger = setup_logging("11_immunotherapy")
    cfg = load_config()
    paths = init_paths()
    rds_dir = paths["rds"]
    data_dir = paths["data"]
    
    logger.info("Evaluating Immunotherapy Response dataset (GSE78220)...")
    
    ess_genes_path = os.path.join(rds_dir, "ess_coefficients.csv")
    if not os.path.exists(ess_genes_path):
        logger.error("Model not locked. Run 04_feature_selection.py first.")
        return
        
    ess_coefs = pd.read_csv(ess_genes_path, index_col=0).iloc[:, 0]
    
    process_gse78220(data_dir, ess_coefs, rds_dir)
    logger.info("Immunotherapy Response evaluation complete.")

if __name__ == "__main__":
    main()
