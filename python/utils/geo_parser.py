import pandas as pd
import gzip
import io
import urllib.request
import logging
import os

logger = logging.getLogger("GEOParser")

def fetch_and_parse_geo(geo_id, dest_dir):
    """
    Downloads and parses a GEO Series Matrix directly over HTTPS.
    Bypasses GEOparse FTP hangs.
    """
    file_name = f"{geo_id}_series_matrix.txt.gz"
    dest_path = os.path.join(dest_dir, file_name)
    
    if not os.path.exists(dest_path):
        # Infer the series range directory (e.g., GSE65904 -> GSE65nnn)
        nnn_dir = geo_id[:-3] + "nnn"
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{nnn_dir}/{geo_id}/matrix/{file_name}"
        logger.info(f"Downloading {url} to {dest_path}")
        urllib.request.urlretrieve(url, dest_path)
        
    logger.info(f"Parsing {dest_path}")
    
    pheno = {}
    expr_lines = []
    in_matrix = False
    
    with gzip.open(dest_path, 'rt') as f:
        for line in f:
            if line.startswith('!Sample_'):
                parts = line.strip().split('\t')
                key = parts[0]
                values = [p.strip('"') for p in parts[1:]]
                
                if key in pheno:
                    # In some GEO matrices, Sample_characteristics_ch1 is repeated
                    # We will append them
                    pheno[key] = [f"{old} | {new}" for old, new in zip(pheno[key], values)]
                else:
                    pheno[key] = values
            elif line.startswith('!series_matrix_table_begin'):
                in_matrix = True
            elif line.startswith('!series_matrix_table_end'):
                in_matrix = False
            elif in_matrix:
                expr_lines.append(line)
                
    pheno_df = pd.DataFrame(pheno)
    if not pheno_df.empty and '!Sample_geo_accession' in pheno_df.columns:
        pheno_df.index = pheno_df['!Sample_geo_accession']
    elif not pheno_df.empty and '!Sample_title' in pheno_df.columns:
        pheno_df.index = pheno_df['!Sample_title']
        
    expr_df = pd.DataFrame()
    if expr_lines:
        expr_df = pd.read_csv(io.StringIO(''.join(expr_lines)), sep='\t', index_col=0)
        
    return expr_df, pheno_df
