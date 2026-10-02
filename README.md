# Epigenetic State → Cancer Phenotype → Therapeutic Vulnerability (Python Port)

A reproducible multi-cohort, multi-omics computational analysis identifying epigenetic cancer cell states and their associated therapeutic vulnerabilities.

## Python Port
This repository has been entirely ported from R to Python. It utilizes modern Python data science libraries including `pandas`, `scikit-learn`, `scikit-survival`, `lifelines`, and `gseapy`.

## Central Question
> Can integrated epigenetic states identify reproducible cancer cell states and their associated therapeutic vulnerabilities across independent patient cohorts?

## Cohort Architecture

| Role | Dataset | Source | Data Type |
|---|---|---|---|
| Discovery | TCGA-SKCM | UCSC Xena | RNA-seq + 450K methylation + clinical |
| Validation | GSE65904 / GSE22153 | GEO | Expression + survival |
| Validation | GSE144487 | GEO | EPIC methylation + survival |
| Immunotherapy | GSE91061 / GSE78220 | GEO | RNA-seq + anti-PD-1 response |
| Functional | DepMap | depmap.org | CRISPR dependency (Chronos) |
| Pharmacological | GDSC2 / PRISM | cancerrxgene | Imputed IC50 |

## Quick Start

### 1. Install Requirements
Create a virtual environment (recommended) and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline
The master orchestrator runs all steps sequentially from data download to figure generation.
```bash
python python/run_pipeline.py
```

## Project Structure
```
Methylation/
├── config.yaml                  # Core configuration
├── requirements.txt             # Python dependencies
├── python/
│   ├── run_pipeline.py          # Master orchestrator
│   ├── 01_data_acquisition.py   # UCSC Xena download
│   ├── 02_qc_preprocessing.py   # Variance filtering / Imputation
│   ├── 03_methylation_expression.py # Spearman correlations
│   ├── 04_feature_selection.py  # LASSO Cox (Model Lock)
│   ├── 05_clustering.py         # Consensus Clustering
│   ├── 06_pathway_enrichment.py # gseapy (ssGSEA/Prerank)
│   ├── 07_immune_deconvolution.py
│   ├── 08_stemness.py
│   ├── 09_survival.py           # lifelines Cox/KM
│   ├── 10_external_validation.py # GEOparse
│   ├── 11_immunotherapy.py
│   ├── 13_depmap.py
│   ├── 14_drug_response.py      # RidgeCV (oncoPredict clone)
│   ├── 15_figures.py            # seaborn/matplotlib
│   └── utils/
│       ├── helpers.py
│       └── gene_sets.py
```

## Methodology Highlights
1. **Model Lock:** Feature coefficients are mathematically locked in script 04 using `scikit-survival`'s LASSO Cox model. 
2. **Data Acquisition:** We utilize UCSC Xena for heavily pre-processed TCGA matrices, avoiding the complex raw data wrangling required by GDC.
3. **Reproducibility:** A centralized `config.yaml` dictates thresholds and seeds (`seed: 42`).
