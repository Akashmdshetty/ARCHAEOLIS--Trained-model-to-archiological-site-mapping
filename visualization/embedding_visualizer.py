import pandas as pd
import yaml
import os
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler

def visualize_embeddings():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    cluster_path = "data/processed/clusters.csv"
    if not os.path.exists(cluster_path):
        print("Error: Clusters file not found. Run discover_sites.py first.")
        return

    df = pd.read_csv(cluster_path)
    features = df.drop(['filename', 'kmeans_cluster', 'dbscan_cluster'], axis=1).values
    
    # Scale
    features_scaled = StandardScaler().fit_transform(features)

    print("Performing UMAP projection...")
    reducer = umap.UMAP(
        n_neighbors=config['visualization']['umap_n_neighbors'],
        min_dist=config['visualization']['umap_min_dist'],
        random_state=42
    )
    embedding = reducer.fit_transform(features_scaled)

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=df['kmeans_cluster'], cmap='Spectral', s=20)
    plt.colorbar(scatter, label='KMeans Cluster')
    plt.title('UMAP Projection of Archaeological Terrain Embeddings')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    
    os.makedirs("visualization", exist_ok=True)
    plt.savefig("visualization/umap_projection.png")
    print("UMAP visualization saved to visualization/umap_projection.png")
    plt.close()

if __name__ == "__main__":
    visualize_embeddings()
