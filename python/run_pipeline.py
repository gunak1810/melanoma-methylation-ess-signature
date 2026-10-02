import os
import sys
import subprocess
import time
import logging
from utils.helpers import load_config, setup_logging

def main():
    logger = setup_logging("run_pipeline")
    cfg = load_config()
    
    print("="*70)
    print("  EPIGENETIC STATE -> CANCER PHENOTYPE -> THERAPEUTIC VULNERABILITY  ")
    print("  Language: Python")
    print("="*70)
    
    scripts = [
        "01_data_acquisition.py",
        "02_qc_preprocessing.py",
        "03_methylation_expression.py",
        "04_feature_selection.py",
        "05_clustering.py",
        "06_pathway_enrichment.py",
        "07_immune_deconvolution.py",
        "08_stemness.py",
        "09_survival.py",
        "10_external_validation.py",
        "11_immunotherapy.py",
        "13_depmap.py",
        "14_drug_response.py",
        "15_figures.py"
    ]
    
    start_time = time.time()
    
    for i, script in enumerate(scripts, start=1):
        script_path = os.path.join("python", script)
        logger.info(f"\n[{i}/{len(scripts)}] Running {script}...")
        
        if not os.path.exists(script_path):
            logger.warning(f"Script not found: {script_path}. Skipping.")
            continue
            
        try:
            result = subprocess.run([sys.executable, script_path], 
                                    check=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed at {script}. Exit code: {e.returncode}")
            sys.exit(1)
            
    elapsed = (time.time() - start_time) / 60
    logger.info(f"\nPipeline completed in {elapsed:.1f} minutes.")

if __name__ == "__main__":
    main()
