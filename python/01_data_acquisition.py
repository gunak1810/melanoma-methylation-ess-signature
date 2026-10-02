import os
import pandas as pd
import requests
import gzip
import shutil
import logging
from utils.helpers import load_config, init_paths, setup_logging

def download_xena_dataset(dataset_id, output_path, base_url="https://tcga.xenahubs.net/download/"):
    """Download a dataset from UCSC Xena Hub if it doesn't exist."""
    if os.path.exists(output_path):
        logging.info(f"Dataset already downloaded: {output_path}")
        return output_path
        
    url_base = f"{base_url}{dataset_id}"
    
    # Try with .gz first (most matrices)
    url_gz = f"{url_base}.gz"
    temp_file = output_path + ".download"
    
    try:
        logging.info(f"Trying to download {url_gz}...")
        r = requests.get(url_gz, stream=True)
        r.raise_for_status()
        is_gz = True
    except requests.exceptions.HTTPError:
        logging.info(f"GZ failed, trying {url_base}...")
        r = requests.get(url_base, stream=True)
        r.raise_for_status()
        is_gz = False
        
    logging.info(f"Downloading to {temp_file}")
    with open(temp_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            
    if is_gz:
        logging.info(f"Extracting {temp_file} to {output_path}")
        with gzip.open(temp_file, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(temp_file)
    else:
        os.rename(temp_file, output_path)
        
    return output_path

def main():
    logger = setup_logging("01_data_acquisition")
    cfg = load_config()
    paths = init_paths()
    
    data_dir = paths["data"]
    rds_dir = paths["rds"]
    
    # TCGA-SKCM Xena IDs
    rna_id = cfg["tcga"]["rna_xena"]
    meth_id = cfg["tcga"]["meth_xena"]
    clin_id = cfg["tcga"]["clinical_xena"]
    mut_id = cfg["tcga"]["mutation_xena"]
    
    # 1. Download and parse RNA-seq
    logger.info("--- Processing RNA-seq ---")
    rna_path = os.path.join(data_dir, "TCGA_SKCM_RNA.tsv")
    download_xena_dataset(rna_id, rna_path)
    
    # Xena RNA-seq is usually log2(x+1). We'll load it.
    if not os.path.exists(os.path.join(rds_dir, "tcga_rna.parquet")):
        logger.info("Loading RNA-seq matrix...")
        rna_df = pd.read_csv(rna_path, sep="\t", index_col=0)
        rna_df.to_parquet(os.path.join(rds_dir, "tcga_rna.parquet"))
        logger.info(f"RNA-seq shape: {rna_df.shape}")
    
    # 2. Download and parse Methylation
    logger.info("--- Processing 450K Methylation ---")
    meth_path = os.path.join(data_dir, "TCGA_SKCM_Meth450.tsv")
    download_xena_dataset(meth_id, meth_path)
    
    if not os.path.exists(os.path.join(rds_dir, "tcga_meth.parquet")):
        logger.info("Loading Methylation matrix (this takes memory/time)...")
        meth_df = pd.read_csv(meth_path, sep="\t", index_col=0)
        meth_df.to_parquet(os.path.join(rds_dir, "tcga_meth.parquet"))
        logger.info(f"Methylation shape: {meth_df.shape}")
        
    # 3. Clinical Data
    logger.info("--- Processing Clinical Data ---")
    clin_path = os.path.join(data_dir, "TCGA_SKCM_Clinical.tsv")
    download_xena_dataset(clin_id, clin_path)
    
    if not os.path.exists(os.path.join(rds_dir, "tcga_clinical.parquet")):
        clin_df = pd.read_csv(clin_path, sep="\t", index_col=0)
        
        # Harmonize OS
        # Xena variables usually look like _OS, _OS_IND
        if "days_to_death" in clin_df.columns and "days_to_last_followup" in clin_df.columns:
            d_death = pd.to_numeric(clin_df["days_to_death"], errors="coerce")
            d_last = pd.to_numeric(clin_df["days_to_last_followup"], errors="coerce")
            clin_df["OS_time"] = d_death.fillna(d_last)
            clin_df["OS_event"] = clin_df["_OS_IND"]
            
        clin_df.to_parquet(os.path.join(rds_dir, "tcga_clinical.parquet"))
        logger.info(f"Clinical shape: {clin_df.shape}")
        
    logger.info("Data acquisition complete.")

if __name__ == "__main__":
    main()
