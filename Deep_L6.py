# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
import scipy.cluster.hierarchy as sch

# Configure plotting style
sns.set_theme(style="whitegrid")
%matplotlib inline
# Load the wine dataset
wine = load_wine()
X = wine.data
y_true = wine.target # True labels (useful for evaluation/comparison)
feature_names = wine.feature_names

# Convert to DataFrame for easier inspection
df_wine = pd.DataFrame(X, columns=feature_names)
df_wine['Target'] = y_true

print(f"Dataset shape: {df_wine.shape}")
df_wine.head()
# Scale features to have mean=0 and variance=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensions to 2D using PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance ratio by 2 PCA components: {pca.explained_variance_ratio_.sum():.2f}")
# Initialize and fit K-Means model (k=3 based on known classes)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Plot K-Means clustering results
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', edgecolor='k', s=50)
plt.title("K-Means Clustering (k=3) on Wine Dataset (PCA-reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster')
plt.show()
# Plot Dendrogram to visualize hierarchical groupings
plt.figure(figsize=(10, 5))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distances")
plt.axhline(y=20, color='r', linestyle='--') # Threshold guide
plt.show()

# Fit Agglomerative Clustering with 3 clusters
agg_clustering = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
agg_labels = agg_clustering.fit_predict(X_scaled)

# Plot Hierarchical Clustering results
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels, cmap='plasma', edgecolor='k', s=50)
plt.title("Agglomerative Hierarchical Clustering (k=3)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster')
plt.show()

# Calculate evaluation metrics
metrics_data = {
    "Model": ["K-Means", "Hierarchical Clustering"],
    "Silhouette Score": [
        silhouette_score(X_scaled, kmeans_labels),
        silhouette_score(X_scaled, agg_labels)
    ],
    "Adjusted Rand Index (ARI)": [
        adjusted_rand_score(y_true, kmeans_labels),
        adjusted_rand_score(y_true, agg_labels)
    ]
}

df_metrics = pd.DataFrame(metrics_data)
df_metrics.set_index("Model", inplace=True)
df_metrics
