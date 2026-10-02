# =============================================================================
# cptac_validation.py — CPTAC Proteomic Validation
# =============================================================================
# Tests whether epigenetic state genes show RNA–protein concordance.
# Uses the cptac Python package to access CPTAC-CM (cutaneous melanoma).
#
# Inputs:
#   ESS gene list (from results/rds/ess_genes.rds → exported as CSV)
#
# Outputs:
#   results/tables/cptac_rna_protein_correlation.csv
#   results/tables/cptac_summary.csv
# =============================================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats

# --- Configuration ---
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
RDS_DIR = os.path.join(RESULTS_DIR, "rds")
os.makedirs(TABLES_DIR, exist_ok=True)

print("=" * 70)
print("  STEP 12: CPTAC PROTEOMIC VALIDATION")
print("=" * 70)

# --- 1. Load ESS genes (exported as CSV from R) ---
ess_genes_file = os.path.join(TABLES_DIR, "ess_genes_for_python.csv")
if not os.path.exists(ess_genes_file):
    print(f"\nWARNING: {ess_genes_file} not found.")
    print("Run the following in R first:")
    print('  ess_genes <- load_rds("ess_genes")')
    print(f'  write.csv(data.frame(gene = ess_genes), "{ess_genes_file}", row.names=FALSE)')
    print("\nAlternatively, enter ESS genes manually.")
    sys.exit(1)

ess_genes = pd.read_csv(ess_genes_file)["gene"].tolist()
print(f"\nLoaded {len(ess_genes)} ESS genes")

# --- 2. Load CPTAC Data ---
try:
    import cptac
    print("\nDownloading CPTAC cutaneous melanoma dataset...")
    cptac.download(dataset="cm", version="latest")
    cm = cptac.Cm()
    print("CPTAC-CM loaded successfully.")
except ImportError:
    print("ERROR: cptac package not installed.")
    print("Install with: pip install cptac")
    sys.exit(1)
except Exception as e:
    print(f"ERROR loading CPTAC data: {e}")
    print("This may be due to network issues or dataset availability.")
    print("Continuing with available data...")
    sys.exit(1)

# --- 3. Get Proteomics and Transcriptomics ---
try:
    proteomics = cm.get_proteomics()
    transcriptomics = cm.get_transcriptomics()
    clinical = cm.get_clinical()

    print(f"\nProteomics shape: {proteomics.shape}")
    print(f"Transcriptomics shape: {transcriptomics.shape}")
    print(f"Clinical shape: {clinical.shape}")
except Exception as e:
    print(f"ERROR accessing data: {e}")
    sys.exit(1)

# --- 4. RNA–Protein Correlation for ESS Genes ---
print("\n--- Computing RNA–Protein Correlations ---")

# Find common patients
common_patients = proteomics.index.intersection(transcriptomics.index)
print(f"Common patients (both RNA + protein): {len(common_patients)}")

# Handle multi-level column names (cptac uses multi-index)
def get_gene_column(df, gene):
    """Find the column for a given gene, handling multi-level columns."""
    if isinstance(df.columns, pd.MultiIndex):
        # Try to find gene in the first level
        matching = [col for col in df.columns if gene in str(col[0])]
        if matching:
            return matching[0]
    else:
        if gene in df.columns:
            return gene
    return None

results = []

for gene in ess_genes:
    prot_col = get_gene_column(proteomics, gene)
    rna_col = get_gene_column(transcriptomics, gene)

    if prot_col is None or rna_col is None:
        continue

    prot_vals = proteomics.loc[common_patients, prot_col]
    rna_vals = transcriptomics.loc[common_patients, rna_col]

    # Handle Series vs DataFrame
    if isinstance(prot_vals, pd.DataFrame):
        prot_vals = prot_vals.iloc[:, 0]
    if isinstance(rna_vals, pd.DataFrame):
        rna_vals = rna_vals.iloc[:, 0]

    # Drop NAs
    valid = prot_vals.notna() & rna_vals.notna()
    n_valid = valid.sum()

    if n_valid < 10:
        continue

    rho, pval = stats.spearmanr(rna_vals[valid], prot_vals[valid])

    results.append({
        "gene": gene,
        "spearman_rho": rho,
        "pvalue": pval,
        "n_samples": n_valid
    })

    print(f"  {gene}: ρ={rho:.3f}, p={pval:.2e}, n={n_valid}")

# --- 5. Compile and Save Results ---
if results:
    results_df = pd.DataFrame(results)

    # FDR correction
    from statsmodels.stats.multitest import multipletests
    _, fdr, _, _ = multipletests(results_df["pvalue"], method="fdr_bh")
    results_df["fdr"] = fdr
    results_df = results_df.sort_values("pvalue")

    results_df.to_csv(os.path.join(TABLES_DIR, "cptac_rna_protein_correlation.csv"),
                       index=False)
    print(f"\nResults saved: {len(results_df)} genes tested")

    # Summary statistics
    n_significant = (results_df["fdr"] < 0.05).sum()
    mean_rho = results_df["spearman_rho"].mean()
    median_rho = results_df["spearman_rho"].median()

    summary = {
        "total_ess_genes": len(ess_genes),
        "genes_in_both": len(results_df),
        "significant_fdr05": n_significant,
        "mean_rho": mean_rho,
        "median_rho": median_rho,
        "fraction_positive_rho": (results_df["spearman_rho"] > 0).mean()
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(TABLES_DIR, "cptac_summary.csv"), index=False)

    print(f"\n--- CPTAC Validation Summary ---")
    print(f"  ESS genes tested: {len(results_df)} / {len(ess_genes)}")
    print(f"  Significant (FDR < 0.05): {n_significant}")
    print(f"  Mean ρ: {mean_rho:.3f}")
    print(f"  Median ρ: {median_rho:.3f}")
    print(f"  Fraction positive ρ: {(results_df['spearman_rho'] > 0).mean():.2f}")

    # Wilcoxon test: are ρ values significantly > 0?
    if len(results_df) > 5:
        stat, pval = stats.wilcoxon(results_df["spearman_rho"])
        print(f"  Wilcoxon signed-rank (ρ > 0): p = {pval:.2e}")
else:
    print("\nNo ESS genes found in both RNA and protein data.")
    print("This may indicate the CPTAC-CM dataset does not cover these genes.")

print("\n" + "=" * 70)
print("  CPTAC Validation Complete")
print("=" * 70)
