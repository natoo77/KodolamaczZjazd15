from tqdm import tqdm

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


class KMeansClusterer:
    def __init__(self, n_clusters: int, batch_size: int):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.model = None
        self.best_params_ = None

    def _create_model(self, n_clusters):
        return MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=self.batch_size,
            random_state=42,
        )

    def find_optimal_clusters(self, X, cluster_range=range(4, 13)):
        scores = []
        for n_clusters in tqdm(cluster_range):
            model = self._create_model(n_clusters)
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels)
            scores.append((n_clusters, score))

        self.best_params_ = max(scores, key=lambda x: x[1])
        return scores

    def fit(self, X):
        if self.best_params_ is None:
            self.model = self._create_model(self.n_clusters)
        else:
            self.model = self._create_model(self.best_params_[0])

        self.model.fit(X)
        return self

    def get_cluster_centers(self):
        return self.model.cluster_centers_

    def get_labels(self, X):
        return self.model.predict(X)

    def get_silhouette(self, X):
        labels = self.get_labels(X)
        return silhouette_score(X, labels)

    def evaluate_clustering(self, X):
        # TODO
        pass