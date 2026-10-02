import os
import pandas as pd
import numpy as np
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from utils.helpers import load_config, init_paths, setup_logging

def consensus_clustering(X, max_k=8, reps=100, p_item=0.8):
    """
    Simplified consensus clustering implementation.
    X: samples x features
    """
    n_samples = X.shape[0]
    
    # Store consensus matrices for each K
    consensus_matrices = {}
    
    for k in range(2, max_k + 1):
        # Co-occurrence matrix
        co_occur = np.zeros((n_samples, n_samples))
        # Counts matrix (how many times a pair was sampled together)
        counts = np.zeros((n_samples, n_samples))
        
        for _ in range(reps):
            # Subsample
            sample_idx = np.random.choice(n_samples, size=int(n_samples * p_item), replace=False)
            X_sub = X[sample_idx]
            
            # Cluster
            clusterer = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
            labels = clusterer.fit_predict(X_sub)
            
            # Update matrices
            for i in range(len(sample_idx)):
                for j in range(i, len(sample_idx)):
                    idx_i = sample_idx[i]
                    idx_j = sample_idx[j]
                    
                    counts[idx_i, idx_j] += 1
                    counts[idx_j, idx_i] += 1
                    
                    if labels[i] == labels[j]:
                        co_occur[idx_i, idx_j] += 1
                        co_occur[idx_j, idx_i] += 1
                        
        # Calculate consensus matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            consensus = co_occur / counts
            consensus = np.nan_to_num(consensus)
            
        consensus_matrices[k] = consensus
        
    return consensus_matrices

def main():
    logger = setup_logging("05_clustering")
    cfg = load_config()
    paths = init_paths()
    rds_dir = paths["rds"]
    
    logger.info("Loading data for clustering...")
    expr_df = pd.read_parquet(os.path.join(rds_dir, "expr_matrix.parquet"))
    clin_df = pd.read_parquet(os.path.join(rds_dir, "tcga_clinical_with_ess.parquet"))
    reg_genes_path = os.path.join(rds_dir, "meth_expr_regulatory_genes.csv")
    
    if os.path.exists(reg_genes_path):
        reg_genes = pd.read_csv(reg_genes_path).iloc[:, 0].tolist()
    else:
        logger.error("Regulatory genes not found.")
        return
        
    # Align
    common_patients = expr_df.columns.intersection(clin_df.index)
    expr_df = expr_df[common_patients]
    
    cluster_genes = [g for g in reg_genes if g in expr_df.index]
    logger.info(f"Clustering on {len(cluster_genes)} regulatory genes...")
    
    X = expr_df.loc[cluster_genes].T.values # samples x genes
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    
    # 1. Consensus Clustering
    max_k = cfg["clustering"]["max_k"]
    reps = cfg["clustering"]["reps"]
    
    logger.info(f"Running consensus clustering (Max K={max_k}, Reps={reps})...")
    consensus_matrices = consensus_clustering(X_scaled, max_k=max_k, reps=reps)
    
    # 2. Select Optimal K (PAC metric)
    logger.info("Computing clustering metrics (PAC, Silhouette)...")
    metrics = []
    
    best_k = 3 # default
    min_pac = float('inf')
    
    for k in range(2, max_k + 1):
        consensus = consensus_matrices[k]
        
        # Proportion of Ambiguous Clustering (PAC)
        # Fraction of entries between 0.1 and 0.9
        pac = np.sum((consensus > 0.1) & (consensus < 0.9)) / (consensus.shape[0] * (consensus.shape[0] - 1))
        
        # Get final cluster assignments by clustering the distance matrix (1 - consensus)
        distance = 1 - consensus
        clusterer = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
        labels = clusterer.fit_predict(distance)
        
        # Silhouette on original scaled data
        sil = silhouette_score(X_scaled, labels)
        
        metrics.append({"k": k, "pac": pac, "silhouette": sil})
        logger.info(f"  K={k}: PAC={pac:.3f}, Silhouette={sil:.3f}")
        
        if k >= 3 and pac < min_pac:
            min_pac = pac
            best_k = k
            
    logger.info(f"Selected Optimal K: {best_k}")
    
    # Final labels for optimal K
    best_consensus = consensus_matrices[best_k]
    best_distance = 1 - best_consensus
    clusterer = AgglomerativeClustering(n_clusters=best_k, metric='precomputed', linkage='average')
    final_labels = clusterer.fit_predict(best_distance)
    
    # Assign states (State_A, State_B, etc)
    state_names = [f"State_{chr(65+i)}" for i in final_labels]
    states_series = pd.Series(state_names, index=common_patients)
    
    # Save assignments
    states_series.to_csv(os.path.join(rds_dir, "patient_states.csv"))
    clin_df["epigenetic_state"] = states_series
    clin_df.to_parquet(os.path.join(rds_dir, "tcga_clinical_with_states.parquet"))
    
    # 3. Dimensionality Reduction
    logger.info("Computing PCA and t-SNE coordinates...")
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X_scaled)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=cfg["project"]["seed"])
    tsne_coords = tsne.fit_transform(X_scaled)
    
    dim_df = pd.DataFrame({
        "patient": common_patients,
        "state": states_series.values,
        "PC1": pca_coords[:, 0],
        "PC2": pca_coords[:, 1],
        "tSNE1": tsne_coords[:, 0],
        "tSNE2": tsne_coords[:, 1]
    })
    
    dim_df.to_csv(os.path.join(rds_dir, "dim_reduction_coords.csv"), index=False)
    
    logger.info("Clustering complete.")

if __name__ == "__main__":
    main()
