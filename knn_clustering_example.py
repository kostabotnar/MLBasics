import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score, confusion_matrix
import os

# Create output directory
output_dir = r"C:\Dev\MLBasics\build\knn-example"
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Create sample dataset with 100 records - SAME as SVM example
n_samples = 100

# Parameter 1: Continuous variable (gene expression level)
gene_expression = np.random.normal(5.0, 2.5, n_samples)

# Parameter 2: Age (continuous variable, realistic range for medical study)
age = np.random.normal(50.0, 12.0, n_samples)
age = np.clip(age, 25, 75)  # Clip to realistic age range

# Create target variable (binary outcome) - for comparison with clusters
linear_combination = 1.2 * gene_expression + 0.05 * age - 8.5
disease_probability = 1 / (1 + np.exp(-linear_combination))
disease_status = np.random.binomial(1, disease_probability, n_samples)

# Create DataFrame
data = pd.DataFrame({
    'gene_expression': gene_expression,
    'age': age,
    'disease_status': disease_status
})

print("Dataset Overview:")
print(f"Total samples: {len(data)}")
print(f"Features: gene_expression (continuous), age (continuous)")
print(f"True disease status (for comparison): binary")
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

print(f"\nOriginal feature ranges:")
print(f"Gene expression: {X[:, 0].min():.2f} to {X[:, 0].max():.2f}")
print(f"Age: {X[:, 1].min():.2f} to {X[:, 1].max():.2f}")

print(f"\nScaled feature ranges:")
print(f"Gene expression (scaled): {X_scaled[:, 0].min():.2f} to {X_scaled[:, 0].max():.2f}")
print(f"Age (scaled): {X_scaled[:, 1].min():.2f} to {X_scaled[:, 1].max():.2f}")

# Apply K-Means clustering to create 2 clusters
print("\n" + "="*50)
print("CLUSTERING ANALYSIS")
print("="*50)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
data['cluster'] = cluster_labels

print(f"\nK-Means Clustering Results:")
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
print(f"Cluster 0: {cluster_counts[0]} samples ({cluster_counts[0]/len(data)*100:.1f}%)")
print(f"Cluster 1: {cluster_counts[1]} samples ({cluster_counts[1]/len(data)*100:.1f}%)")

# Calculate clustering metrics
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.3f} (higher is better, range: -1 to 1)")

# Analyze cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print(f"\nCluster Centers (original scale):")
print(f"Cluster 0: Gene Expression = {cluster_centers[0, 0]:.2f}, Age = {cluster_centers[0, 1]:.2f}")
print(f"Cluster 1: Gene Expression = {cluster_centers[1, 0]:.2f}, Age = {cluster_centers[1, 1]:.2f}")

# Compare clusters with actual disease status
print(f"\n" + "="*50)
print("CLUSTER vs DISEASE STATUS COMPARISON")
print("="*50)

# Create contingency table
contingency_table = pd.crosstab(data['cluster'], data['disease_status'], 
                               rownames=['Cluster'], colnames=['Disease Status'])
print(f"\nContingency Table:")
print(contingency_table)

# Calculate adjusted rand score (measures agreement between clusterings)
ari_score = adjusted_rand_score(data['disease_status'], cluster_labels)
print(f"\nAdjusted Rand Index: {ari_score:.3f} (1.0 = perfect agreement, 0.0 = random)")

# Analyze cluster characteristics
print(f"\nCluster Characteristics:")
for cluster_id in [0, 1]:
    cluster_data = data[data['cluster'] == cluster_id]
    print(f"\nCluster {cluster_id}:")
    print(f"  Sample count: {len(cluster_data)}")
    print(f"  Gene expression - Mean: {cluster_data['gene_expression'].mean():.2f}, Std: {cluster_data['gene_expression'].std():.2f}")
    print(f"  Age - Mean: {cluster_data['age'].mean():.2f}, Std: {cluster_data['age'].std():.2f}")
    print(f"  Disease rate: {cluster_data['disease_status'].mean():.3f} ({cluster_data['disease_status'].sum()}/{len(cluster_data)})")

# KNN Analysis for understanding cluster boundaries
print(f"\n" + "="*50)
print("K-NEAREST NEIGHBORS ANALYSIS")
print("="*50)

# Use KNN to understand local neighborhood structure
knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_scaled)

# For each point, find its 5 nearest neighbors
distances, indices = knn.kneighbors(X_scaled)

# Calculate average distance to nearest neighbors for each cluster
avg_distances_by_cluster = {}
for cluster_id in [0, 1]:
    cluster_mask = cluster_labels == cluster_id
    cluster_distances = distances[cluster_mask]
    avg_distances_by_cluster[cluster_id] = cluster_distances.mean()
    print(f"Cluster {cluster_id} - Average distance to 5 nearest neighbors: {avg_distances_by_cluster[cluster_id]:.3f}")

# Analyze cluster purity (how many neighbors belong to same cluster)
cluster_purity = []
for i in range(len(X_scaled)):
    neighbors = indices[i][1:]  # Exclude the point itself
    neighbor_clusters = cluster_labels[neighbors]
    same_cluster = np.sum(neighbor_clusters == cluster_labels[i])
    purity = same_cluster / len(neighbors)
    cluster_purity.append(purity)

data['cluster_purity'] = cluster_purity
print(f"\nCluster Purity Analysis:")
print(f"Average cluster purity: {np.mean(cluster_purity):.3f} (1.0 = all neighbors in same cluster)")
for cluster_id in [0, 1]:
    cluster_purity_avg = data[data['cluster'] == cluster_id]['cluster_purity'].mean()
    print(f"Cluster {cluster_id} average purity: {cluster_purity_avg:.3f}")

# Create visualizations (save only, don't show)
plt.ioff()  # Turn off interactive mode

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('K-Means Clustering Analysis for Bioinformatics Data', fontsize=16)

# Plot 1: Original data colored by true disease status
scatter1 = axes[0, 0].scatter(data['gene_expression'], data['age'], 
                             c=data['disease_status'], cmap='RdBu', alpha=0.7, s=60)
axes[0, 0].set_xlabel('Gene Expression Level')
axes[0, 0].set_ylabel('Age (years)')
axes[0, 0].set_title('Original Data: True Disease Status')
axes[0, 0].grid(True, alpha=0.3)
cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
cbar1.set_label('Disease Status')
cbar1.set_ticks([0, 1])
cbar1.set_ticklabels(['No Disease', 'Disease'])

# Plot 2: Data colored by clusters
scatter2 = axes[0, 1].scatter(data['gene_expression'], data['age'], 
                             c=data['cluster'], cmap='viridis', alpha=0.7, s=60)
# Add cluster centers
axes[0, 1].scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                  c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')
axes[0, 1].set_xlabel('Gene Expression Level')
axes[0, 1].set_ylabel('Age (years)')
axes[0, 1].set_title('K-Means Clustering Results')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
cbar2.set_label('Cluster')
cbar2.set_ticks([0, 1])
cbar2.set_ticklabels(['Cluster 0', 'Cluster 1'])

# Plot 3: Contingency table heatmap
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Cluster vs Disease Status\nContingency Table')
axes[1, 0].set_xlabel('Disease Status')
axes[1, 0].set_ylabel('Cluster')

# Plot 4: Cluster purity visualization
scatter3 = axes[1, 1].scatter(data['gene_expression'], data['age'], 
                             c=data['cluster_purity'], cmap='plasma', alpha=0.7, s=60)
axes[1, 1].set_xlabel('Gene Expression Level')
axes[1, 1].set_ylabel('Age (years)')
axes[1, 1].set_title('Cluster Purity\n(Based on 5 Nearest Neighbors)')
axes[1, 1].grid(True, alpha=0.3)
cbar3 = plt.colorbar(scatter3, ax=axes[1, 1])
cbar3.set_label('Cluster Purity')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'clustering_overview.png'), dpi=300, bbox_inches='tight')
print(f"\nClustering overview saved to {os.path.join(output_dir, 'clustering_overview.png')}")
plt.close()

# Create detailed cluster comparison plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Detailed Cluster Analysis', fontsize=16)

# Plot 1: Side by side comparison
for i, (title, color_col) in enumerate([('True Disease Status', 'disease_status'), 
                                       ('K-Means Clusters', 'cluster')]):
    ax = axes[i]
    scatter = ax.scatter(data['gene_expression'], data['age'], 
                        c=data[color_col], cmap='RdBu' if i == 0 else 'viridis', 
                        alpha=0.7, s=60)
    
    if i == 1:  # Add cluster centers for clustering plot
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                  c='red', marker='x', s=200, linewidths=3)
    
    ax.set_xlabel('Gene Expression Level')
    ax.set_ylabel('Age (years)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    if i == 0:
        cbar.set_label('Disease Status')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['No Disease', 'Disease'])
    else:
        cbar.set_label('Cluster')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Cluster 0', 'Cluster 1'])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cluster_comparison.png'), dpi=300, bbox_inches='tight')
print(f"Cluster comparison saved to {os.path.join(output_dir, 'cluster_comparison.png')}")
plt.close()

# Create KNN neighborhood visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Show a few example points and their nearest neighbors
example_points = [10, 25, 40, 60, 80]  # Indices of example points
colors = ['red', 'green', 'blue', 'orange', 'purple']

# Plot all points
scatter = ax.scatter(data['gene_expression'], data['age'], 
                    c=data['cluster'], cmap='viridis', alpha=0.4, s=40)

# Highlight example points and their neighbors
for i, (point_idx, color) in enumerate(zip(example_points, colors)):
    # Highlight the main point
    ax.scatter(data.iloc[point_idx]['gene_expression'], data.iloc[point_idx]['age'], 
              c=color, s=200, marker='*', edgecolors='black', linewidth=2,
              label=f'Point {point_idx}')
    
    # Highlight its neighbors
    neighbor_indices = indices[point_idx][1:6]  # 5 nearest neighbors
    neighbor_data = data.iloc[neighbor_indices]
    ax.scatter(neighbor_data['gene_expression'], neighbor_data['age'], 
              c=color, alpha=0.7, s=80, marker='o', edgecolors='black')

ax.set_xlabel('Gene Expression Level')
ax.set_ylabel('Age (years)')
ax.set_title('K-Nearest Neighbors Example\n(Stars = Example Points, Circles = Their 5 Nearest Neighbors)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'knn_neighborhoods.png'), dpi=300, bbox_inches='tight')
print(f"KNN neighborhoods visualization saved to {os.path.join(output_dir, 'knn_neighborhoods.png')}")
plt.close()

# Save detailed results
clustering_results = pd.DataFrame({
    'gene_expression': data['gene_expression'],
    'age': data['age'],
    'true_disease_status': data['disease_status'],
    'cluster_assignment': data['cluster'],
    'cluster_purity': data['cluster_purity']
})

# Create summary statistics
summary_stats = pd.DataFrame({
    'metric': ['silhouette_score', 'adjusted_rand_index', 'cluster_0_size', 'cluster_1_size',
               'cluster_0_disease_rate', 'cluster_1_disease_rate', 'avg_cluster_purity'],
    'value': [silhouette_avg, ari_score, cluster_counts[0], cluster_counts[1],
              data[data['cluster'] == 0]['disease_status'].mean(),
              data[data['cluster'] == 1]['disease_status'].mean(),
              np.mean(cluster_purity)]
})

# Save cluster centers
cluster_centers_df = pd.DataFrame(cluster_centers, 
                                 columns=['gene_expression_center', 'age_center'],
                                 index=['cluster_0', 'cluster_1'])

clustering_results.to_csv(os.path.join(output_dir, 'clustering_results.csv'), index=False)
summary_stats.to_csv(os.path.join(output_dir, 'clustering_metrics.csv'), index=False)
cluster_centers_df.to_csv(os.path.join(output_dir, 'cluster_centers.csv'))

print(f"\nResults saved to:")
print(f"- Clustering results: {os.path.join(output_dir, 'clustering_results.csv')}")
print(f"- Clustering metrics: {os.path.join(output_dir, 'clustering_metrics.csv')}")
print(f"- Cluster centers: {os.path.join(output_dir, 'cluster_centers.csv')}")
print(f"- Graphics: clustering_overview.png, cluster_comparison.png, knn_neighborhoods.png")
print(f"- Dataset: dataset.csv")

# Final summary
print(f"\n" + "="*60)
print("CLUSTERING SUMMARY")
print("="*60)
print(f"+ Created 2 clusters from gene expression and age data")
print(f"+ Silhouette Score: {silhouette_avg:.3f} (quality of clustering)")
print(f"+ Adjusted Rand Index: {ari_score:.3f} (agreement with true disease status)")
print(f"+ Average Cluster Purity: {np.mean(cluster_purity):.3f} (neighborhood consistency)")
print(f"+ Cluster 0: {cluster_counts[0]} samples, {data[data['cluster'] == 0]['disease_status'].mean():.3f} disease rate")
print(f"+ Cluster 1: {cluster_counts[1]} samples, {data[data['cluster'] == 1]['disease_status'].mean():.3f} disease rate")

if ari_score > 0.3:
    print(f"\n> GOOD AGREEMENT: Clusters align well with disease status!")
elif ari_score > 0.1:
    print(f"\n> MODERATE AGREEMENT: Clusters partially align with disease status")
else:
    print(f"\n> POOR AGREEMENT: Clusters don't align well with disease status")

# Turn interactive mode back on
plt.ion()