import logging
import numpy as np

from clustering.data.processors import MSNBCDataProcessor
from clustering.models.kmeans import KMeansClusterer
from clustering.visualization import ClusterVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize objects
    processor = MSNBCDataProcessor()
    model = KMeansClusterer(8, 8192)
    visualizer = ClusterVisualizer(processor.category_mapping)

    # Load and preprocess data
    logger.info("Loading and processing data")
    sequences = processor.load_data("../data/msnbc990928.seq")
    X = processor.preprocess_sequences(sequences)
    np.random.shuffle(X)
    X_sampled = X[:30000]

    # Find optimal number of clusters
    logger.info("Finding optimal number of clusters...")
    cluster_scores = model.find_optimal_clusters(X_sampled)

    # Fit clustering model
    logger.info("Fitting clustering model")
    model.fit(X_sampled)

    # Get results
    logger.info("Getting the results")
    centers = model.get_cluster_centers()
    logger.info(f"Example cluster center: {centers[0]}")
    labels = model.get_labels(X_sampled)
    logger.info(f"First 10 labels: {labels[:10]}")

    # Visualize the results
    logger.info("Visualizing results")
    logger.info("Plotting cluster profiles")
    visualizer.plot_cluster_profiles(centers)
    logger.info("Plotting cluster sizes")
    visualizer.plot_cluster_sizes(labels)

    # # Scoring the model
    # logger.info("Getting silhouette score")
    # sil_score = model.get_silhouette(X)
    # logger.info(f"Silhouette Score: {sil_score:.3f}")

if __name__ == "__main__":
    main()