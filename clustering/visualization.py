import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ClusterVisualizer:
    def __init__(self, category_mapping):
        self.categoty_mapping = category_mapping

    def plot_cluster_profiles(self, centers, figsize=(15, 8)):
        plt.figure(figsize=figsize)
        sns.heatmap(
             centers,
             xticklabels=[self.categoty_mapping[i+1] for i in range(17)],
             yticklabels=[f"Cluster {i}" for i in range(centers.shape[0])],
             cmap="YlOrRd"
        )
        plt.title("Cluster Profiles: Category frequencies")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_cluster_sizes(self, labels, figsize=(15, 8)):
        cluster_sizes = pd.Series(labels).value_counts()

        plt.figure(figsize=figsize)
        cluster_sizes.plot(kind="bar")
        plt.title("Cluster Size Distribution")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Sessions")
        plt.tight_layout()
        plt.show()
