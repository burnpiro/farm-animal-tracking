import numpy as np
from abc import ABC
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine, mahalanobis
from collections import deque
from filterpy.kalman import KalmanFilter


class Track:
    def __init__(self, bbox=None, embedding=None, track_id=None) -> None:
        # self.bbox = self.bbox_to_xywa(bbox)        
        self.embedding = embedding
        self.track_id = track_id
        self.history = []

        self.embeddings = deque([embedding], maxlen=10)
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1
        # Linear motion kalman model
        # Dynamic matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        # Initial uncertainty
        self.kf.P = np.array([
            [3, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 10, 0, 0, 0],
            [0, 0, 0, 0, 0, 10, 0, 0],
            [0, 0, 0, 0, 0, 0, 10, 0],
            [0, 0, 0, 0, 0, 0, 0, 10]
        ])
        # Measurement noise covariance
        self.kf.R *= 0.005
        # Process Noise Covariance
        # self.kf.Q[-1, -1] *= 0.001
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.Q *= 0.01

        bbox = Track.bbox_to_xywa(bbox)
        self.kf.x = np.concatenate(
            [bbox, np.zeros_like(bbox)], axis=-1)
        # self.embeddings = [embedding]
        self.VI = None

    @staticmethod
    def bbox_to_xywa(bbox):
        """
        Converts bounding box from format (left, top, width, height) to (left, top, width, height/width)
        """
        box = bbox.copy()
        box[3] = bbox[3]/bbox[2]
        return box

    def predict(self):
        self.kf.predict()
        self.VI = np.linalg.inv(self.kf.P[:4, :4])

    def update(self, bbox):
        bbox = Track.bbox_to_xywa(bbox)
        self.kf.update(bbox)
        self.history.append(bbox)

    def get_position_distance(self, new_bbox):
        bbox = self.kf.x[:4]
        new_bbox = Track.bbox_to_xywa(new_bbox)
        return mahalanobis(new_bbox, bbox, self.VI)

    def get_distance(self, new_bbox, new_embedding, similarity_coeff):
        pos_dist = self.get_position_distance(new_bbox)
        sim_dist = self.get_similarity_distance(new_embedding)

        return (1-similarity_coeff)*pos_dist+similarity_coeff*sim_dist

    def get_similarity_distance(self, new_embedding):
        return np.min([
            cosine(embedding, new_embedding)
            for embedding in self.embeddings
        ])

    # def set_bbox(self, bbox):
    #     self.bbox = self.bbox_to_xywa(bbox)

    def get_bbox(self):
        # box = self.bbox.copy()
        box = self.kf.x[:4].copy()
        box[3] = box[3]*box[2]
        return box


class Tracker(ABC):
    def __init__(self, paths_num) -> None:
        self.paths_num = paths_num
        # self.tracks = [Track() for _ in range(paths_num)]
        self.tracks = None
        self.appearance_weight = 0.8

    @staticmethod
    def boxes_to_xywh(boxes):
        wh = boxes[:, 2:]-boxes[:, :2]
        xy = boxes[:, :2]+wh/2
        return np.concatenate([xy, wh], axis=-1)

    @staticmethod
    def pre_process_boxes(boxes):
        new_boxes = []
        for idx, (box, _) in enumerate(boxes.items()):
            new_boxes.append(box)

        return new_boxes

    def similarity_matrix(self, new_bboxes, new_embeddings):
        matrix = np.empty((self.paths_num, new_embeddings.shape[0]))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i, j] = self.tracks[i].get_distance(
                    new_bboxes[j],
                    new_embeddings[j]
                )
        return matrix

    def position_similarity_matrix(self, new_bboxes):
        matrix = np.empty((self.paths_num, new_bboxes.shape[0]))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i, j] = self.tracks[i].get_position_distance(
                    new_bboxes[j],
                )
        return matrix

    def appearance_similarity_matrix(self, new_embeddings):
        matrix = np.empty((self.paths_num, new_embeddings.shape[0]))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i, j] = self.tracks[i].get_similarity_distance(
                    new_embeddings[j]
                )
        return matrix

    def run(self, boxes, embeddings):
        boxes = Tracker.pre_process_boxes(boxes)
        boxes = np.array(boxes)
        if boxes.shape[0] == 0:
            return None

        boxes = Tracker.boxes_to_xywh(boxes)

        if self.tracks is None:
            if boxes.shape[0] < self.paths_num:
                return None

            self.tracks = [
                Track(boxes[i], embeddings[i], i+1)
                for i in range(self.paths_num)
            ]

        for track in self.tracks:
            track.predict()

        # t_1 = 9.4877
        t_1 = 8
        t_2 = 0.3
        position_similarity_matrix = self.position_similarity_matrix(boxes)
        position_similarity_matrix[position_similarity_matrix > t_1] = t_1

        appearance_similarity_matrix = self.appearance_similarity_matrix(
            embeddings)
        appearance_similarity_matrix[appearance_similarity_matrix > t_2] = t_2

        admissible = np.logical_and(
            position_similarity_matrix < t_1, appearance_similarity_matrix < t_2)
        similarity_matrix = (1-self.appearance_weight) * position_similarity_matrix + \
            self.appearance_weight*appearance_similarity_matrix

        rows, cols = linear_sum_assignment(similarity_matrix)

        for i, j in zip(rows, cols):
            if not admissible[i, j]:
                continue
            # self.tracks[i].embedding = embeddings[j]
            self.tracks[i].embeddings.append(embeddings[j])
            # self.tracks[i].set_bbox(boxes[j])
            self.tracks[i].update(boxes[j])
