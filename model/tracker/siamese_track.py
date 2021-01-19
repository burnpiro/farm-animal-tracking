import numpy as np
from math import sqrt
from scipy.spatial.distance import euclidean
from model.tracker.default_track import DefaultTrack, AbstractTrack


class SiameseTrack(DefaultTrack):
    def __init__(self, track_id=None) -> None:
        super().__init__()
        self.track_id = None
        self.history = None
        self.prev_emb = None

        self.initialize_track(track_id, None)

    def initialize_track(self, track_id=None, prev_emb=None) -> None:
        self.track_id = track_id
        self.prev_emb = prev_emb
        self.history = []

    def update(self, bbox, embedding, **kwargs):
        self.prev_emb = embedding
        self.history.append(bbox)

    def update_with_prev_value(self):
        self.update(self.history[-1], self.prev_emb)

    def get_similarity_to_embedding(self, emb) -> float:
        """
        Calculates distance between given embedding and currently stored
        Args:
            emb: ndarray - object embedding

        Returns: float
            distance between embeddings
        """
        if self.prev_emb is None:
            return len(emb)  # max value is when each element is set to 1

        return euclidean(self.prev_emb, emb)
