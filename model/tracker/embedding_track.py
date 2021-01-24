import numpy as np
from math import sqrt
from scipy.spatial.distance import euclidean
from model.tracker.default_track import DefaultTrack, AbstractTrack

statuses = {"onScreen": 0, "offScreen": 1, "lost": 2}


class EmbeddingTrack(DefaultTrack):
    def __init__(self, track_id=None) -> None:
        super().__init__()
        self.track_id = None
        self.history = None
        self.prev_emb = None
        self.embeddings = []
        self.predicted_class = None
        self.status = statuses["offScreen"]

        self.initialize_track(track_id, None)

    def initialize_track(self, track_id=None, prev_emb=None) -> None:
        self.track_id = track_id
        self.prev_emb = prev_emb
        self.history = []

    def update(self, bbox, embedding, **kwargs):
        if np.all([bbox, [-1, -1, -1, -1]]):
            self.status = statuses["offScreen"]
        else:
            self.status = statuses["onScreen"]
        self.prev_emb = embedding
        self.history.append(bbox)

    def add_embedding(self, embedding):
        self.embeddings.append(embedding)

    def merge_with_track(self, track: DefaultTrack) -> None:
        """
        Appends track onto current track
        Args:
            track: EmbeddingTrack

        Returns:

        """
        for path in track.history:
            self.update(path, track.prev_emb)

    def update_with_prev_value(self):
        self.update(self.history[-1], self.prev_emb)
