import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
import os

# Create output directory
output_dir = r"C:\Dev\MLBasics\build\dbscan-example"
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic dataset that favors DBSCAN - non-spherical clusters with outliers
n_samples = 150

print("Creating synthetic dataset optimized for DBSCAN...")

# Create three distinct cluster types that DBSCAN handles well:
# 1. Dense circular cluster (healthy young patients)
# 2. Elongated cluster (progressive disease patients)
# 3. Scattered outliers (rare cases)

# Cluster 1: Dense circular cluster (50 points)
n_cluster1 = 50
cluster1_gene = np.random.normal(3.0, 0.8, n_cluster1)
cluster1_age = np.random.normal(35.0, 6.0, n_cluster1)
cluster1_disease = np.zeros(n_cluster1)  # Healthy group

# Cluster 2: Elongated/crescent-shaped cluster (60 points) - disease progression
n_cluster2 = 60
# Create crescent shape using parametric equations
t = np.linspace(0, np.pi, n_cluster2)
cluster2_gene_base = 6.0 + 2.5 * np.cos(t) + np.random.normal(0, 0.3, n_cluster2)
cluster2_age_base = 55.0 + 1.5 * np.sin(t) + np.random.normal(0, 2.0, n_cluster2)

# Add some thickness to the crescent
thickness_offset = np.random.normal(0, 0.5, n_cluster2)
cluster2_gene = cluster2_gene_base + thickness_offset * 0.3
cluster2_age = cluster2_age_base + thickness_offset

# Most have disease (progressive disease pattern)
cluster2_disease = np.random.binomial(1, 0.8, n_cluster2)

# Cluster 3: Small dense cluster (high-risk group) - 25 points
n_cluster3 = 25
cluster3_gene = np.random.normal(8.5, 0.6, n_cluster3)
cluster3_age = np.random.normal(65.0, 4.0, n_cluster3)
cluster3_disease = np.ones(n_cluster3)  # All have disease

# Add outliers/noise points (15 points) - rare cases
n_outliers = 15
outlier_gene = np.random.uniform(-1, 10, n_outliers)
outlier_age = np.random.uniform(20, 80, n_outliers)
outlier_disease = np.random.binomial(1, 0.3, n_outliers)  # Mixed disease status

# Combine all data
gene_expression = np.concatenate([cluster1_gene, cluster2_gene, cluster3_gene, outlier_gene])
age = np.concatenate([cluster1_age, cluster2_age, cluster3_age, outlier_age])
disease_status = np.concatenate([cluster1_disease, cluster2_disease, cluster3_disease, outlier_disease])

# Create ground truth cluster labels for evaluation
true_clusters = np.concatenate([
    np.zeros(n_cluster1),      # Cluster 0: Healthy young
    np.ones(n_cluster2),       # Cluster 1: Disease progression
    np.full(n_cluster3, 2),    # Cluster 2: High-risk elderly
    np.full(n_outliers, -1)    # Outliers
])

# Update n_samples
n_samples = len(gene_expression)

print(f"Created {n_samples} samples:")
print(f"- Cluster 1 (Healthy young): {n_cluster1} samples")
print(f"- Cluster 2 (Disease progression): {n_cluster2} samples")
print(f"- Cluster 3 (High-risk elderly): {n_cluster3} samples")
print(f"- Outliers (Rare cases): {n_outliers} samples")

# Create DataFrame
data = pd.DataFrame({
    'gene_expression': gene_expression,
    'age': age,
    'disease_status': disease_status,
    'true_cluster': true_clusters  # Ground truth for evaluation
})

print("Dataset Overview:")
print(f"Total samples: {len(data)}")
print(f"Features: gene_expression (continuous), age (continuous)")
print(f"True disease status (for comparison): binary")
print(f"Ground truth clusters: 3 clusters + outliers (-1)")
print("\nDataset head:")
print(data.head(10))
print("\nDataset summary:")
print(data.describe())

# Show class distribution
disease_counts = data['disease_status'].value_counts()
print(f"\nDisease status distribution:")
print(f"No Disease (0): {disease_counts[0]} samples ({disease_counts[0]/len(data)*100:.1f}%)")
print(f"Disease (1): {disease_counts[1]} samples ({disease_counts[1]/len(data)*100:.1f}%)")

# Save dataset
data.to_csv(os.path.join(output_dir, 'dataset.csv'), index=False)
print(f"\nDataset saved to {os.path.join(output_dir, 'dataset.csv')}")

# Prepare features for clustering (without using disease_status)
X = data[['gene_expression', 'age']].values

# Scale features for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n" + "="*50)
print("DBSCAN PARAMETER OPTIMIZATION")
print("="*50)

# Find optimal eps using k-distance plot
# DBSCAN requires two parameters: eps (neighborhood size) and min_samples
k = 4  # min_samples - 1 (rule of thumb: min_samples = dimensions + 1)
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# Sort distances to k-th nearest neighbor
distances = np.sort(distances[:, k-1], axis=0)

print(f"Finding optimal eps parameter using {k}-distance plot...")

# Try different eps values to find optimal one (wider range for new dataset)
eps_candidates = np.linspace(0.2, 2.0, 25)
best_eps = None
best_score = -1
best_n_clusters = 0
results = []

print(f"Testing eps values from {eps_candidates[0]:.2f} to {eps_candidates[-1]:.2f}...")

for eps in eps_candidates:
    dbscan = DBSCAN(eps=eps, min_samples=4)
    cluster_labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    # Calculate silhouette score only if we have valid clusters
    if n_clusters > 1 and n_noise < len(X_scaled) - 1:
        try:
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        except:
            silhouette_avg = -1
    else:
        silhouette_avg = -1
    
    results.append({
        'eps': eps,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette_score': silhouette_avg
    })
    
    # Select best parameters (prefer 2-3 clusters with good silhouette score)
    if 2 <= n_clusters <= 3 and silhouette_avg > best_score and n_noise < len(X_scaled) * 0.3:
        best_eps = eps
        best_score = silhouette_avg
        best_n_clusters = n_clusters

# If no good eps found, use a reasonable default
if best_eps is None:
    best_eps = 0.5
    print(f"No optimal eps found, using default: {best_eps}")
else:
    print(f"Optimal eps found: {best_eps:.3f} (silhouette score: {best_score:.3f})")

print(f"\n" + "="*50)
print("DBSCAN CLUSTERING ANALYSIS")
print("="*50)

# Apply DBSCAN with optimal parameters
dbscan = DBSCAN(eps=best_eps, min_samples=4)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Add cluster labels to dataframe
data['dbscan_cluster'] = dbscan_labels

# Calculate clustering metrics
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_dbscan = list(dbscan_labels).count(-1)

print(f"DBSCAN Results:")
print(f"Number of clusters: {n_clusters_dbscan}")
print(f"Number of noise points: {n_noise_dbscan} ({n_noise_dbscan/len(data)*100:.1f}%)")
print(f"Parameters used: eps={best_eps:.3f}, min_samples=4")

# Analyze cluster distribution
if n_clusters_dbscan > 0:
    cluster_counts_dbscan = pd.Series(dbscan_labels).value_counts().sort_index()
    print(f"\nCluster distribution:")
    for cluster_id in sorted(set(dbscan_labels)):
        if cluster_id == -1:
            print(f"Noise points: {cluster_counts_dbscan[cluster_id]} samples ({cluster_counts_dbscan[cluster_id]/len(data)*100:.1f}%)")
        else:
            print(f"Cluster {cluster_id}: {cluster_counts_dbscan[cluster_id]} samples ({cluster_counts_dbscan[cluster_id]/len(data)*100:.1f}%)")

# Calculate silhouette score (excluding noise points)
if n_clusters_dbscan > 1:
    non_noise_mask = dbscan_labels != -1
    if np.sum(non_noise_mask) > 1:
        try:
            silhouette_dbscan = silhouette_score(X_scaled[non_noise_mask], dbscan_labels[non_noise_mask])
            print(f"Silhouette Score (excluding noise): {silhouette_dbscan:.3f}")
        except:
            print(f"Could not calculate silhouette score")
            silhouette_dbscan = None
    else:
        silhouette_dbscan = None
else:
    silhouette_dbscan = None

# Compare with K-Means clustering (for reference)
print(f"\n" + "="*50)
print("COMPARISON WITH K-MEANS")
print("="*50)

# Run K-Means for comparison (use 3 clusters to match ground truth)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
data['kmeans_cluster'] = kmeans_labels

print(f"K-Means Results (k=3):")
kmeans_counts = pd.Series(kmeans_labels).value_counts().sort_index()
for cluster_id in sorted(set(kmeans_labels)):
    print(f"Cluster {cluster_id}: {kmeans_counts[cluster_id]} samples ({kmeans_counts[cluster_id]/len(data)*100:.1f}%)")

kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
print(f"K-Means Silhouette Score: {kmeans_silhouette:.3f}")

# Compare with ground truth clusters and disease status
print(f"\n" + "="*50)
print("CLUSTER vs GROUND TRUTH COMPARISON")
print("="*50)

# DBSCAN vs Ground Truth Clusters
if n_clusters_dbscan > 0:
    print(f"DBSCAN vs Ground Truth Clusters:")
    dbscan_gt_contingency = pd.crosstab(data['dbscan_cluster'], data['true_cluster'], 
                                       rownames=['DBSCAN Cluster'], colnames=['True Cluster'])
    print(dbscan_gt_contingency)
    
    dbscan_gt_ari = adjusted_rand_score(data['true_cluster'], dbscan_labels)
    print(f"DBSCAN vs Ground Truth ARI: {dbscan_gt_ari:.3f}")

# K-Means vs Ground Truth Clusters  
print(f"\nK-Means vs Ground Truth Clusters:")
kmeans_gt_contingency = pd.crosstab(data['kmeans_cluster'], data['true_cluster'], 
                                   rownames=['K-Means Cluster'], colnames=['True Cluster'])
print(kmeans_gt_contingency)

kmeans_gt_ari = adjusted_rand_score(data['true_cluster'], kmeans_labels)
print(f"K-Means vs Ground Truth ARI: {kmeans_gt_ari:.3f}")

print(f"\n" + "="*50)
print("CLUSTER vs DISEASE STATUS COMPARISON")
print("="*50)

# DBSCAN vs Disease Status
if n_clusters_dbscan > 0:
    print(f"DBSCAN Contingency Table:")
    dbscan_contingency = pd.crosstab(data['dbscan_cluster'], data['disease_status'], 
                                   rownames=['DBSCAN Cluster'], colnames=['Disease Status'])
    print(dbscan_contingency)
    
    dbscan_ari = adjusted_rand_score(data['disease_status'], dbscan_labels)
    print(f"DBSCAN vs Disease Status ARI: {dbscan_ari:.3f}")

# K-Means vs Disease Status
print(f"\nK-Means Contingency Table:")
kmeans_contingency = pd.crosstab(data['kmeans_cluster'], data['disease_status'], 
                               rownames=['K-Means Cluster'], colnames=['Disease Status'])
print(kmeans_contingency)

kmeans_ari = adjusted_rand_score(data['disease_status'], kmeans_labels)
print(f"K-Means vs Disease Status ARI: {kmeans_ari:.3f}")

# Analyze cluster characteristics
print(f"\n" + "="*50)
print("CLUSTER CHARACTERISTICS")
print("="*50)

# DBSCAN cluster analysis
if n_clusters_dbscan > 0:
    print(f"DBSCAN Cluster Characteristics:")
    for cluster_id in sorted(set(dbscan_labels)):
        cluster_data = data[data['dbscan_cluster'] == cluster_id]
        if cluster_id == -1:
            print(f"\nNoise Points:")
        else:
            print(f"\nDBSCAN Cluster {cluster_id}:")
        print(f"  Sample count: {len(cluster_data)}")
        print(f"  Gene expression - Mean: {cluster_data['gene_expression'].mean():.2f}, Std: {cluster_data['gene_expression'].std():.2f}")
        print(f"  Age - Mean: {cluster_data['age'].mean():.2f}, Std: {cluster_data['age'].std():.2f}")
        if len(cluster_data) > 0:
            print(f"  Disease rate: {cluster_data['disease_status'].mean():.3f} ({cluster_data['disease_status'].sum()}/{len(cluster_data)})")

# Create visualizations (save only, don't show)
plt.ioff()  # Turn off interactive mode

# Create comprehensive DBSCAN visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('DBSCAN Clustering Analysis for Bioinformatics Data', fontsize=16)

# Plot 1: Original data colored by ground truth clusters
# Create color map for ground truth (including outliers)
gt_colors = data['true_cluster'].copy()
# Map outliers (-1) to a distinct value for coloring
gt_colors_mapped = np.where(gt_colors == -1, 3, gt_colors)

scatter1 = axes[0, 0].scatter(data['gene_expression'], data['age'], 
                             c=gt_colors_mapped, cmap='tab10', alpha=0.7, s=60)
axes[0, 0].set_xlabel('Gene Expression Level')
axes[0, 0].set_ylabel('Age (years)')
axes[0, 0].set_title('Ground Truth: Designed Clusters\n(Healthy, Disease Progression, High-Risk, Outliers)')
axes[0, 0].grid(True, alpha=0.3)

# Add legend for ground truth
handles1, labels1 = scatter1.legend_elements()
axes[0, 0].legend(handles1, ['Healthy Young', 'Disease Progression', 'High-Risk Elderly', 'Outliers'])

# Plot 2: DBSCAN results
if n_clusters_dbscan > 0:
    # Use a colormap that includes a distinct color for noise (-1)
    unique_labels = sorted(set(dbscan_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            # Noise points in black
            mask = dbscan_labels == label
            axes[0, 1].scatter(data.loc[mask, 'gene_expression'], data.loc[mask, 'age'], 
                             c='black', marker='x', alpha=0.7, s=60, label='Noise')
        else:
            mask = dbscan_labels == label
            axes[0, 1].scatter(data.loc[mask, 'gene_expression'], data.loc[mask, 'age'], 
                             c=[colors[i]], alpha=0.7, s=60, label=f'Cluster {label}')

axes[0, 1].set_xlabel('Gene Expression Level')
axes[0, 1].set_ylabel('Age (years)')
axes[0, 1].set_title(f'DBSCAN Results\n(eps={best_eps:.3f}, min_samples=4)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: K-Means results for comparison
scatter3 = axes[0, 2].scatter(data['gene_expression'], data['age'], 
                             c=data['kmeans_cluster'], cmap='viridis', alpha=0.7, s=60)
# Add K-means cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
axes[0, 2].scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                  c='red', marker='x', s=200, linewidths=3, label='Centroids')
axes[0, 2].set_xlabel('Gene Expression Level')
axes[0, 2].set_ylabel('Age (years)')
axes[0, 2].set_title('K-Means Results (k=2)')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Parameter exploration results
eps_values = [r['eps'] for r in results]
n_clusters_values = [r['n_clusters'] for r in results]
silhouette_values = [r['silhouette_score'] if r['silhouette_score'] != -1 else np.nan for r in results]

axes[1, 0].plot(eps_values, n_clusters_values, 'bo-', label='Number of Clusters')
axes[1, 0].axvline(x=best_eps, color='red', linestyle='--', label=f'Chosen eps={best_eps:.3f}')
axes[1, 0].set_xlabel('eps parameter')
axes[1, 0].set_ylabel('Number of Clusters')
axes[1, 0].set_title('DBSCAN Parameter Exploration')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: DBSCAN vs K-Means contingency heatmaps
if n_clusters_dbscan > 0:
    sns.heatmap(dbscan_contingency, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_title('DBSCAN vs Disease Status')
    axes[1, 1].set_xlabel('Disease Status')
    axes[1, 1].set_ylabel('DBSCAN Cluster')

# Plot 6: Method comparison
methods = ['DBSCAN', 'K-Means']
# Use ground truth ARI for comparison (more meaningful for this synthetic dataset)
if silhouette_dbscan is not None:
    silhouette_scores = [silhouette_dbscan, kmeans_silhouette]
    ari_scores = [dbscan_gt_ari, kmeans_gt_ari]  # Compare with ground truth
else:
    silhouette_scores = [0, kmeans_silhouette]  # Use 0 if DBSCAN silhouette not available
    ari_scores = [dbscan_gt_ari if n_clusters_dbscan > 0 else 0, kmeans_gt_ari]

x = np.arange(len(methods))
width = 0.35

bars1 = axes[1, 2].bar(x - width/2, silhouette_scores, width, label='Silhouette Score', alpha=0.8)
bars2 = axes[1, 2].bar(x + width/2, ari_scores, width, label='Adjusted Rand Index', alpha=0.8)

axes[1, 2].set_xlabel('Clustering Method')
axes[1, 2].set_ylabel('Score')
axes[1, 2].set_title('Method Comparison')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(methods)
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    if not np.isnan(height):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')

for bar in bars2:
    height = bar.get_height()
    if not np.isnan(height):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'dbscan_overview.png'), dpi=300, bbox_inches='tight')
print(f"\nDBSCAN overview saved to {os.path.join(output_dir, 'dbscan_overview.png')}")
plt.close()

# Create k-distance plot for parameter selection
plt.figure(figsize=(10, 6))
plt.plot(range(len(distances)), distances, 'b-')
plt.axhline(y=best_eps, color='red', linestyle='--', label=f'Chosen eps = {best_eps:.3f}')
plt.xlabel('Points (sorted by distance)')
plt.ylabel(f'{k}-th Nearest Neighbor Distance')
plt.title('K-Distance Plot for DBSCAN Parameter Selection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'k_distance_plot.png'), dpi=300, bbox_inches='tight')
print(f"K-distance plot saved to {os.path.join(output_dir, 'k_distance_plot.png')}")
plt.close()

# Save detailed results
dbscan_results = pd.DataFrame({
    'gene_expression': data['gene_expression'],
    'age': data['age'],
    'true_cluster': data['true_cluster'],
    'true_disease_status': data['disease_status'],
    'dbscan_cluster': data['dbscan_cluster'],
    'kmeans_cluster': data['kmeans_cluster']
})

# Create parameter exploration results
param_exploration_df = pd.DataFrame(results)

# Create summary statistics
summary_stats = pd.DataFrame({
    'metric': ['dbscan_n_clusters', 'dbscan_n_noise', 'dbscan_silhouette_score', 
               'dbscan_vs_ground_truth_ari', 'dbscan_vs_disease_ari',
               'kmeans_silhouette_score', 'kmeans_vs_ground_truth_ari', 'kmeans_vs_disease_ari', 
               'optimal_eps', 'min_samples'],
    'value': [n_clusters_dbscan, n_noise_dbscan, 
              silhouette_dbscan if silhouette_dbscan is not None else np.nan, 
              dbscan_gt_ari if n_clusters_dbscan > 0 else np.nan,
              dbscan_ari if n_clusters_dbscan > 0 else np.nan,
              kmeans_silhouette, kmeans_gt_ari, kmeans_ari, best_eps, 4]
})

# Save all results
dbscan_results.to_csv(os.path.join(output_dir, 'dbscan_results.csv'), index=False)
param_exploration_df.to_csv(os.path.join(output_dir, 'parameter_exploration.csv'), index=False)
summary_stats.to_csv(os.path.join(output_dir, 'clustering_metrics.csv'), index=False)

print(f"\nResults saved to:")
print(f"- DBSCAN results: {os.path.join(output_dir, 'dbscan_results.csv')}")
print(f"- Parameter exploration: {os.path.join(output_dir, 'parameter_exploration.csv')}")
print(f"- Clustering metrics: {os.path.join(output_dir, 'clustering_metrics.csv')}")
print(f"- Graphics: dbscan_overview.png, k_distance_plot.png")
print(f"- Dataset: dataset.csv")

# Final summary
print(f"\n" + "="*60)
print("DBSCAN CLUSTERING SUMMARY")
print("="*60)
print(f"+ DBSCAN found {n_clusters_dbscan} clusters with {n_noise_dbscan} noise points")
print(f"+ Parameters: eps={best_eps:.3f}, min_samples=4")
if silhouette_dbscan is not None:
    print(f"+ DBSCAN Silhouette Score: {silhouette_dbscan:.3f}")
if n_clusters_dbscan > 0:
    print(f"+ DBSCAN vs Ground Truth ARI: {dbscan_gt_ari:.3f}")
    print(f"+ DBSCAN vs Disease Status ARI: {dbscan_ari:.3f}")
print(f"+ K-Means Silhouette Score: {kmeans_silhouette:.3f} (for comparison)")
print(f"+ K-Means vs Ground Truth ARI: {kmeans_gt_ari:.3f} (for comparison)")
print(f"+ K-Means vs Disease Status ARI: {kmeans_ari:.3f} (for comparison)")

# Interpretation
if n_clusters_dbscan == 0:
    print(f"\n> DBSCAN found no distinct clusters - data may be too uniform")
elif n_clusters_dbscan == 1:
    print(f"\n> DBSCAN found one large cluster - data is relatively homogeneous")
elif n_noise_dbscan > len(data) * 0.3:
    print(f"\n> DBSCAN found many noise points - data may be very scattered")
else:
    print(f"\n> DBSCAN successfully identified distinct density-based clusters")
    
if n_clusters_dbscan > 0 and silhouette_dbscan is not None:
    if dbscan_gt_ari > kmeans_gt_ari:
        print(f"+ DBSCAN performed better than K-Means for this dataset (ARI: {dbscan_gt_ari:.3f} vs {kmeans_gt_ari:.3f})")
    else:
        print(f"+ K-Means performed better than DBSCAN for this dataset (ARI: {kmeans_gt_ari:.3f} vs {dbscan_gt_ari:.3f})")
        
print(f"\nDataset Design Notes:")
print(f"+ This synthetic dataset was designed to favor DBSCAN with:")
print(f"  - Non-spherical clusters (crescent shape)")
print(f"  - Varying cluster densities")
print(f"  - Natural outliers")
print(f"  - Arbitrary cluster shapes that K-Means struggles with")

# Turn interactive mode back on
plt.ion()