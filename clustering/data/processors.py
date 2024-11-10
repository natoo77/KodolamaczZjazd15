from collections import Counter
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler


class MSNBCDataProcessor:
    def __init__(self):
        self.category_mapping = {
            1: 'frontpage', 2: 'news', 3: 'tech', 4: 'local',
            5: 'opinion', 6: 'on-air', 7: 'misc', 8: 'weather',
            9: 'health', 10: 'living', 11: 'business', 12: 'sports',
            13: 'summary', 14: 'bbs', 15: 'travel', 16: 'msn-news',
            17: 'msn-sports'
        }
        self.preprocessor = self._create_preprocessor()

    def _create_preprocessor(self):
        frequency_transformer = Pipeline([
            ("scaler", StandardScaler()),
            ("normalize", PowerTransformer(method="yeo-johnson")),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("frequencies", frequency_transformer, list(range(17)))
            ],
            remainder="passthrough",
        )
        return preprocessor

    def load_data(self, file_path: str | None = None) -> list[list[int]]:
        """

        :param file_path:
        :return:
        """
        if file_path is None:
            file_path = "data/msnbc990928.seq"
        file_path = Path(file_path)

        sequences = []
        with file_path.open("r") as f:
            for line in f:
                if not line.startswith("%"):
                    sequence = [int(x) for x in line.strip().split()]
                    if len(sequence):
                        sequences.append(sequence)

        return sequences


    def create_feature_vector(self, sequence: list[int]) -> list[int]:
        """

        :param sequence:
        :return:
        """
        counts = Counter(sequence)
        return [counts.get(i, 0) for i in range(1, len(self.category_mapping) + 1)]

    def preprocess_sequences(self, sequences: list[list[int]], min_sequence_length: int = 3, max_category_ratio: float = 0.9):
        # Filtering
        filtered_sequences = []

        for seq in sequences:
            # Filtering too short sequences
            if len(seq) < min_sequence_length:
                continue

            # Filtering single dominant categories
            category_counts = Counter(seq)
            most_common_count = category_counts.most_common(1)[0][1]
            if most_common_count / len(seq) > max_category_ratio:
                continue

            filtered_sequences.append(seq)

        print(f"Reduced the number of sequences from {len(sequences)} to {len(filtered_sequences)}")

        # Create feature vectors
        feature_vectors = [
            self.create_feature_vector(seq)
            for seq in sequences
        ]
        X = np.array(feature_vectors)
        assert X.shape[1] == len(self.category_mapping)

        X_transformed = self.preprocessor.fit_transform(X)

        return X_transformed