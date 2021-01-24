import numpy as np


from scipy.spatial.distance import cosine, euclidean
from collections import deque
from model.tracker.abstract_classes import AbstractTrack


class Track(AbstractTrack):

    def __init__(self, bbox=None, embedding=None, track_id=None) -> None:
        super().__init__()
        self.embedding = None
        self.track_id = None
        self.history = None

        self.embeddings = None
        self.bbox = None

        self.initialize_track(bbox, embedding, track_id)

    def initialize_track(self, bbox=None, embedding=None, track_id=None) -> None:
        self.embedding = embedding
        self.track_id = track_id
        self.bbox = self.bbox_to_xywa(bbox) if self.bbox else np.array([0, 0, 0, 1])
        self.history = [bbox.tolist()]
        self.embeddings = deque(maxlen=50) if embedding is None else deque([embedding], maxlen=50)

    @staticmethod
    def bbox_to_xywa(bbox):
        """
        Converts bounding box from format (left, top, width, height) to (left, top, width, height/width)
        """
        box = bbox.copy()
        box[3] = bbox[3] / bbox[2]
        return box

    def update_with_prev_value(self):
        self.update(self.bbox, None)

    def update(self, bbox, **kwargs):
        bbox = Track.bbox_to_xywa(bbox)
        self.bbox = bbox
        self.history.append(self.bbox[:2].tolist())

    def get_position_distance(self, new_bbox):
        bbox = self.bbox
        new_bbox = Track.bbox_to_xywa(new_bbox)
        return euclidean(new_bbox, bbox)

    def get_distance(self, new_bbox, new_embedding, similarity_coeff):
        pos_dist = self.get_position_distance(new_bbox)
        sim_dist = self.get_similarity_distance(new_embedding)

        return (1 - similarity_coeff) * pos_dist + similarity_coeff * sim_dist

    def get_similarity_distance(self, new_embedding):
        return np.mean(
            [cosine(embedding, new_embedding) for embedding in self.embeddings]
        )

    def get_history(self):
        return [h[:2][::-1] for h in self.history]

    def get_bbox(self):
        box = self.bbox.copy()
        box[3] = box[3] * box[2]
        return box
