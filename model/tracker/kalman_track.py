import numpy as np


from scipy.spatial.distance import cosine, mahalanobis
from collections import deque
from filterpy.kalman import KalmanFilter
from model.tracker.abstract_classes import AbstractTrack


class KalmanTrack(AbstractTrack):

    def __init__(self, bbox=None, embedding=None, track_id=None) -> None:
        super().__init__()
        self.embedding = None
        self.track_id = None
        self.history = None

        self.embeddings = None
        self.kf = None
        self.VI = None
        self.initialize_track(bbox, embedding, track_id)

    def initialize_track(self, bbox=None, embedding=None, track_id=None) -> None:
        # self.bbox = self.bbox_to_xywa(bbox)
        self.embedding = embedding
        self.track_id = track_id
        if bbox is None:
            bbox = np.array([0, 0, 0.1, 0.1])
        self.history = [bbox.tolist()]

        self.embeddings = deque(maxlen=50) if embedding is None else deque([embedding], maxlen=50)
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1
        # Linear motion kalman model
        # Dynamic matrix
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, dt, 0, 0, 0],
                [0, 1, 0, 0, 0, dt, 0, 0],
                [0, 0, 1, 0, 0, 0, dt, 0],
                [0, 0, 0, 1, 0, 0, 0, dt],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        # Measurement matrix
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )
        # Initial uncertainty
        self.kf.P = np.array(
            [
                [3, 0, 0, 0, 0, 0, 0, 0],
                [0, 3, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 3, 0, 0, 0, 0],
                [0, 0, 0, 0, 10, 0, 0, 0],
                [0, 0, 0, 0, 0, 10, 0, 0],
                [0, 0, 0, 0, 0, 0, 10, 0],
                [0, 0, 0, 0, 0, 0, 0, 10],
            ]
        )
        # Measurement noise covariance
        self.kf.R *= 0.005
        # Process Noise Covariance
        # self.kf.Q[-1, -1] *= 0.001
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.Q *= 0.01

        bbox = KalmanTrack.bbox_to_xywa(bbox)
        self.kf.x = np.concatenate([bbox, np.zeros_like(bbox)], axis=-1)
        # self.embeddings = [embedding]
        self.VI = None

    def update_with_prev_value(self):
        self.update(self.history[-1])

    @staticmethod
    def bbox_to_xywa(bbox):
        """
        Converts bounding box from format (left, top, width, height) to (left, top, width, height/width)
        """
        box = bbox.copy()
        box[3] = bbox[3] / bbox[2]
        return box

    # @staticmethod
    # def xywa_to_bbox(bbox):
    #     """
    #     Converts bounding box from format (left, top, width, height) to (left, top, width, height/width)
    #     """
    #     box = bbox.copy()
    #     box[3] = bbox[3] * bbox[2]
    #     box[2:] += box[:2]
    #     return box

    def predict(self):
        self.kf.predict()
        self.VI = np.linalg.inv(self.kf.P[:4, :4])
        self.history.append(self.kf.x[:2].tolist())

    def update(self, bbox):
        bbox = KalmanTrack.bbox_to_xywa(bbox)
        self.kf.update(bbox)

    def get_position_distance(self, new_bbox):
        bbox = self.kf.x[:4]
        new_bbox = KalmanTrack.bbox_to_xywa(new_bbox)
        return mahalanobis(new_bbox, bbox, self.VI)

    def get_distance(self, new_bbox, new_embedding, similarity_coeff):
        pos_dist = self.get_position_distance(new_bbox)
        sim_dist = self.get_similarity_distance(new_embedding)

        return (1 - similarity_coeff) * pos_dist + similarity_coeff * sim_dist

    def get_similarity_distance(self, new_embedding):
        return np.min(
            [cosine(embedding, new_embedding) for embedding in self.embeddings]
        )

    # def set_bbox(self, bbox):
    #     self.bbox = self.bbox_to_xywa(bbox)

    def get_history(self):
        return [h[:2][::-1] for h in self.history]

    def get_bbox(self):
        # box = self.bbox.copy()
        box = self.kf.x[:4].copy()
        box[3] = box[3] * box[2]
        return box
