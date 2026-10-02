import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from utils.helpers import load_config, init_paths, setup_logging
from utils.geo_parser import fetch_and_parse_geo

def plot_km(time, event, scores, title, outpath, median_split=True):
    plt.figure(figsize=(5, 5))
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    
    med = np.median(scores)
    high_idx = scores > med
    low_idx = scores <= med
    
    kmf_high.fit(time[high_idx], event[high_idx], label="High ESS")
    kmf_low.fit(time[low_idx], event[low_idx], label="Low ESS")
    
    ax = plt.subplot(111)
    kmf_high.plot_survival_function(ax=ax, color="red")
    kmf_low.plot_survival_function(ax=ax, color="blue")
    
    plt.title(title)
    plt.xlabel("Time (Days)")
    plt.ylabel("Survival Probability")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def main():
    logger = setup_logging("generate_final_figures")
    cfg = load_config()
    paths = init_paths()
    rds_dir = paths["rds"]
    fig_dir = paths["figures"]
    
    sns.set_theme(style="ticks", context="paper")
    
    # 1. Internal Validation KM
    test_surv_path = os.path.join(rds_dir, "test_survival.csv")
    if os.path.exists(test_surv_path):
        df = pd.read_csv(test_surv_path)
        plot_km(df["OS_time"], df["OS_event"], df["ESS"], 
                "TCGA-SKCM Internal Test Set (N=138)\nKaplan-Meier Survival", 
                os.path.join(fig_dir, "Fig1A_TCGA_KM.png"))
                
    # 2. Scatterplots
    expr_path = os.path.join(rds_dir, "expr_matrix.parquet")
    meth_path = os.path.join(rds_dir, "meth_matrix.parquet")
    pairs_path = os.path.join(rds_dir, "meth_expr_pairs_sig.parquet")
    
    if os.path.exists(expr_path) and os.path.exists(meth_path) and os.path.exists(pairs_path):
        expr_df = pd.read_parquet(expr_path)
        meth_df = pd.read_parquet(meth_path)
        pairs_df = pd.read_parquet(pairs_path)
        
        # Select key genes for scatterplots
        key_genes = ["SLFN12", "HLA-DOB", "EN2", "CDH3", "PKP1", "FERMT3"]
        common_samples = expr_df.columns.intersection(meth_df.columns)
        
        for gene in key_genes:
            gene_pairs = pairs_df[pairs_df["gene"] == gene]
            if not gene_pairs.empty:
                best_cpg = gene_pairs.sort_values("abs_rho", ascending=False).iloc[0]["cpg"]
                rho = gene_pairs.sort_values("abs_rho", ascending=False).iloc[0]["rho"]
                
                plt.figure(figsize=(4, 4))
                x = meth_df.loc[best_cpg, common_samples]
                y = expr_df.loc[gene, common_samples]
                
                sns.regplot(x=x, y=y, scatter_kws={'alpha':0.5, 's':10}, line_kws={'color':'red'})
                plt.title(f"{gene} Expression vs {best_cpg} Methylation\n$\\rho$ = {rho:.2f}")
                plt.xlabel("DNA Methylation ($\\beta$-value)")
                plt.ylabel("RNA Expression (log2 FPKM+1)")
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"Fig2_{gene}_Scatter.png"), dpi=300)
                plt.close()
                
    # 3. TSNE
    tsne_path = os.path.join(rds_dir, "dim_reduction_coords.csv")
    if os.path.exists(tsne_path):
        dim_df = pd.read_csv(tsne_path)
        plt.figure(figsize=(5, 4))
        sns.scatterplot(data=dim_df, x="tSNE1", y="tSNE2", hue="state", palette="Set2", alpha=0.8, edgecolor="none")
        plt.title("Patient Epigenetic States (t-SNE)")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "Fig3_TSNE_States.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
