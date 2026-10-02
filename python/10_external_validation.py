import os
import pandas as pd
import numpy as np
import logging
import urllib.request
import gzip
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from utils.helpers import load_config, init_paths, setup_logging
from utils.geo_parser import fetch_and_parse_geo

def fetch_gpl_annotation(gpl_id, data_dir):
    logger = logging.getLogger("10_external_validation")
    dest = os.path.join(data_dir, f"{gpl_id}.annot.gz")
    if not os.path.exists(dest):
        nnn = gpl_id[:-3] + "nnn"
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{nnn}/{gpl_id}/annot/{gpl_id}.annot.gz"
        logger.info(f"Downloading GPL annotation {url}")
        urllib.request.urlretrieve(url, dest)
        
    lines = []
    with gzip.open(dest, 'rt', encoding='utf8') as f:
        for line in f:
            if not line.startswith('^') and not line.startswith('#') and not line.startswith('!'):
                lines.append(line)
                
    import io
    df = pd.read_csv(io.StringIO(''.join(lines)), sep='\t', usecols=["ID", "Gene symbol"])
    df = df.dropna(subset=["Gene symbol"])
    df["Gene symbol"] = df["Gene symbol"].str.split("///").str[0].str.strip()
    return df.set_index("ID")["Gene symbol"]

def validate_cohort(geo_id, gpl_id, ess_coefs, data_dir, rds_dir, parse_fn):
    """Generic validation function. parse_fn extracts (os_time, os_event) from pheno."""
    logger = logging.getLogger("10_external_validation")
    logger.info(f"Processing {geo_id}...")
    
    expr, pheno = fetch_and_parse_geo(geo_id, data_dir)
    logger.info(f"Downloaded {expr.shape[0]} probes x {expr.shape[1]} samples")
    
    probe_to_gene = fetch_gpl_annotation(gpl_id, data_dir)
    
    expr.index = expr.index.map(probe_to_gene)
    expr = expr[expr.index.notna()]
    expr = expr.groupby(expr.index).mean()
    
    logger.info(f"Mapped to {expr.shape[0]} unique genes.")
    
    # Calculate ESS
    missing_genes = [g for g in ess_coefs.index if g not in expr.index]
    if missing_genes:
        logger.warning(f"Missing {len(missing_genes)} ESS genes in {geo_id}: {missing_genes}")
        
    common = [g for g in ess_coefs.index if g in expr.index]
    ess_scores = expr.loc[common].T.dot(ess_coefs.loc[common].values)
    
    # Parse Survival using the cohort-specific function
    os_time, os_event = parse_fn(pheno)
    
    surv = pd.DataFrame({"ESS": ess_scores, "time": os_time, "event": os_event}).dropna()
    surv = surv[surv["time"] > 0]
    logger.info(f"Parsed survival for {surv.shape[0]} samples.")
    
    if surv.empty or surv.shape[0] < 10:
        logger.error(f"Insufficient survival data for {geo_id}")
        return
        
    # Logrank
    median_ess = surv["ESS"].median()
    high = surv[surv["ESS"] > median_ess]
    low = surv[surv["ESS"] <= median_ess]
    
    res = logrank_test(high["time"], low["time"], high["event"], low["event"])
    logger.info(f"{geo_id} Log-Rank p-value: {res.p_value:.4f}")
    
    # Cox
    try:
        cph = CoxPHFitter()
        cph.fit(surv[["time", "event", "ESS"]], duration_col="time", event_col="event")
        hr = cph.summary.loc["ESS", "exp(coef)"]
        p = cph.summary.loc["ESS", "p"]
        ci_lo = cph.summary.loc["ESS", "exp(coef) lower 95%"]
        ci_hi = cph.summary.loc["ESS", "exp(coef) upper 95%"]
        logger.info(f"{geo_id} Cox: HR={hr:.2f} (95%CI: {ci_lo:.2f}-{ci_hi:.2f}), p={p:.4f}")
    except Exception as e:
        logger.error(f"Cox failed for {geo_id}: {e}")


def parse_gse65904(pheno):
    """GSE65904: disease specific survival in days / disease specific survival (1=death, 0=alive)"""
    os_time = pd.Series(index=pheno.index, dtype=float)
    os_event = pd.Series(index=pheno.index, dtype=float)
    
    char_cols = [c for c in pheno.columns if 'characteristics' in c]
    for sample in pheno.index:
        t = np.nan
        e = np.nan
        for c in char_cols:
            full_val = str(pheno.loc[sample, c])
            for val in full_val.split('|'):
                val_lower = val.strip().lower()
                # Disease-specific survival time
                if 'disease specific survival in days' in val_lower:
                    try:
                        t = float(val.split(':')[-1].strip())
                    except:
                        pass
                # Disease-specific survival event
                elif 'disease specific survival' in val_lower and ('death' in val_lower or 'alive' in val_lower):
                    s = val.split(':')[-1].strip()
                    try:
                        e = float(s)
                    except:
                        pass
        os_time[sample] = t
        os_event[sample] = e
    return os_time, os_event


def parse_gse19234(pheno):
    """GSE19234: days lived since metastasis / staus dead or alive: 0/1"""
    os_time = pd.Series(index=pheno.index, dtype=float)
    os_event = pd.Series(index=pheno.index, dtype=float)
    
    char_cols = [c for c in pheno.columns if 'characteristics' in c]
    for sample in pheno.index:
        t = np.nan
        e = np.nan
        for c in char_cols:
            full_val = str(pheno.loc[sample, c])
            for val in full_val.split('|'):
                val_lower = val.strip().lower()
                if 'days lived since metastasis' in val_lower:
                    try:
                        t = float(val.split(':')[-1].strip())
                    except:
                        pass
                elif 'staus dead or alive' in val_lower or 'status dead or alive' in val_lower:
                    s = val.split(':')[-1].strip()
                    try:
                        e = float(s)
                    except:
                        pass
        os_time[sample] = t
        os_event[sample] = e
    return os_time, os_event


def main():
    logger = setup_logging("10_external_validation")
    cfg = load_config()
    paths = init_paths()
    rds_dir = paths["rds"]
    data_dir = paths["data"]
    
    logger.info("Loading frozen ESS model...")
    ess_genes_path = os.path.join(rds_dir, "ess_coefficients.csv")
    if not os.path.exists(ess_genes_path):
        logger.error("Model not locked. Run 04_feature_selection.py first.")
        return
        
    ess_coefs = pd.read_csv(ess_genes_path, index_col=0).iloc[:, 0]
    
    # Cohort 1: GSE65904 (Illumina HumanHT-12, n=214, disease-specific survival)
    logger.info("=" * 50)
    logger.info("COHORT 1: GSE65904 (Bogunovic/Cirenajwis, n=214)")
    logger.info("=" * 50)
    validate_cohort("GSE65904", "GPL10558", ess_coefs, data_dir, rds_dir, parse_gse65904)
    
    # Cohort 2: GSE19234 (Affymetrix HG-U133, n=44, overall survival)
    logger.info("=" * 50)
    logger.info("COHORT 2: GSE19234 (Bogunovic et al., n=44)")
    logger.info("=" * 50)
    validate_cohort("GSE19234", "GPL570", ess_coefs, data_dir, rds_dir, parse_gse19234)
    
    logger.info("External Validation complete.")

if __name__ == "__main__":
    main()
