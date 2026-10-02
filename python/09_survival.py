import os
import pandas as pd
import numpy as np
import logging
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from utils.helpers import load_config, init_paths, setup_logging

def main():
    logger = setup_logging("09_survival")
    cfg = load_config()
    paths = init_paths()
    rds_dir = paths["rds"]
    
    logger.info("Loading clinical data with states and ESS...")
    clin_df = pd.read_parquet(os.path.join(rds_dir, "tcga_clinical_with_states.parquet"))
    
    # Load Test Samples
    test_samples_path = os.path.join(rds_dir, "test_samples.csv")
    if os.path.exists(test_samples_path):
        test_samples = pd.read_csv(test_samples_path)["Sample"].tolist()
        from utils.helpers import get_tcga_patient_id
        # Convert test_samples to 12-char patient IDs to match clin_df.index
        test_patients = [get_tcga_patient_id(s) for s in test_samples]
        test_patients = [p for p in test_patients if p in clin_df.index]
        clin_df = clin_df.loc[test_patients]
        logger.info(f"Subsetting to {len(test_patients)} internal test set patients...")
    else:
        logger.error("test_samples.csv not found. Did you run 03/04?")
        return
    
    # Require survival data
    valid_surv = clin_df["OS_time"].notna() & clin_df["OS_event"].notna() & (clin_df["OS_time"] > 0)
    surv_df = clin_df[valid_surv].copy()
    logger.info(f"Test Set patients with valid survival: {surv_df.shape[0]}")
    
    # 1. ESS Survival Analysis (High vs Low)
    logger.info("Running KM analysis for ESS Groups...")
    
    # Logrank test
    high = surv_df[surv_df["ESS_group"] == "High"]
    low = surv_df[surv_df["ESS_group"] == "Low"]
    
    if len(high) > 0 and len(low) > 0:
        results = logrank_test(high["OS_time"], low["OS_time"], 
                               event_observed_A=high["OS_event"], 
                               event_observed_B=low["OS_event"])
        logger.info(f"ESS High vs Low Log-rank p-value: {results.p_value:.2e}")
    
    # Cox PH for continuous ESS
    logger.info("Running Univariate Cox for continuous ESS...")
    cph_ess = CoxPHFitter()
    try:
        cph_ess.fit(surv_df[["OS_time", "OS_event", "ESS"]], duration_col="OS_time", event_col="OS_event")
        cph_ess.print_summary()
    except Exception as e:
        logger.error(f"Univariate Cox failed: {e}")
    
    # 2. State Survival Analysis
    logger.info("Running KM analysis for Epigenetic States...")
    if "epigenetic_state" in surv_df.columns:
        state_df = surv_df[["OS_time", "OS_event", "epigenetic_state"]].dropna().copy()
        state_dummies = pd.get_dummies(state_df["epigenetic_state"], drop_first=True)
        cox_data = pd.concat([state_df[["OS_time", "OS_event"]], state_dummies], axis=1)
        
        cph_state = CoxPHFitter()
        try:
            cph_state.fit(cox_data, duration_col="OS_time", event_col="OS_event")
            cph_state.print_summary()
        except Exception as e:
            logger.error(f"State Cox failed: {e}")
        
    # 3. Multivariable Cox Model
    logger.info("Running Multivariable Cox Model...")
    
    covariates = ["ESS"]
    multi_df = surv_df[["OS_time", "OS_event", "ESS"]].copy()
    
    if "age_at_initial_pathologic_diagnosis" in surv_df.columns:
        multi_df["Age"] = surv_df["age_at_initial_pathologic_diagnosis"]
        covariates.append("Age")
        
    if "gender" in surv_df.columns:
        multi_df["Male"] = (surv_df["gender"] == "MALE").astype(float)
        covariates.append("Male")
        
    if "pathologic_stage" in surv_df.columns:
        stage_map = {"Stage I": 1, "Stage IA": 1, "Stage IB": 1,
                     "Stage II": 2, "Stage IIA": 2, "Stage IIB": 2, "Stage IIC": 2,
                     "Stage III": 3, "Stage IIIA": 3, "Stage IIIB": 3, "Stage IIIC": 3,
                     "Stage IV": 4}
        multi_df["Stage"] = surv_df["pathologic_stage"].map(stage_map)
        covariates.append("Stage")
        
    multi_df = multi_df.dropna()
    logger.info(f"Multivariable model N={multi_df.shape[0]} patients")
    
    if len(covariates) > 1 and len(multi_df) > 10:
        cph_multi = CoxPHFitter()
        try:
            cph_multi.fit(multi_df, duration_col="OS_time", event_col="OS_event")
            logger.info("Multivariable Model Summary:")
            cph_multi.print_summary()
            
            summary_df = cph_multi.summary
            summary_df.to_csv(os.path.join(rds_dir, "multivariable_cox.csv"))
        except Exception as e:
            logger.error(f"Multivariable Cox failed: {e}")
        
    logger.info("Survival Analysis complete.")

if __name__ == "__main__":
    main()
