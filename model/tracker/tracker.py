from model.tracker.kalman_track import KalmanTrack
import numpy as np
from abc import ABC
from scipy.optimize import linear_sum_assignment

from helpers.bb_helper import pre_process_boxes
from model.tracker.track import Track
from model.tracker.kalman_track import KalmanTrack
from model.tracker.abstract_classes import AbstractTracker


class Tracker(AbstractTracker):
    '''
    Tracker that takes positions and appearances into account while tracking.
    Has 2 variants:
    - normal: uses euclidean distance
    - deepsort: uses kalman filter and mahalanobis distance    
    '''

    def __init__(self, paths_num, appearance_weight=0.4, deepsort=True, max_mahalanobis_distance=9.4877, max_euclidean_distance=10) -> None:
        super().__init__()
        self.steps = 0
        self.appearance_weight = appearance_weight
        self.paths_num = paths_num
        self.reset_tracker()
        if deepsort:
            self.distance_cutoff = max_mahalanobis_distance
            self.track_class = KalmanTrack
        else:
            self.distance_cutoff = max_euclidean_distance
            self.track_class = Track

    def reset_tracker(self) -> None:
        """
            Initializes tracker and resets tracks
        Returns: None

        """
        # self.tracks = [Track() for _ in range(paths_num)]
        self.steps = 0
        self.tracks = None

    def skip_empty_frame(self):
        for track_id, track in enumerate(self.tracks):
            track.update_with_prev_value()

    def get_track_class(self, track):
        return track.track_id

    @staticmethod
    def boxes_to_xywh(boxes):
        wh = boxes[:, 2:] - boxes[:, :2]
        xy = boxes[:, :2] + wh / 2
        return np.concatenate([xy, wh], axis=-1)

    def get_history(self):
        """

        Returns:
            Dict<object_id, List<(x,y)>>
            Object with list of positions for every tracking object
        """
        history = {track.track_id: track.get_history()
                   for track in self.tracks}
        # print(history)
        return history

    def similarity_matrix(self, new_bboxes, new_embeddings):
        matrix = np.empty((self.paths_num, new_embeddings.shape[0]))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i, j] = self.tracks[i].get_distance(
                    new_bboxes[j], new_embeddings[j]
                )
        return matrix

    def position_similarity_matrix(self, new_bboxes, max_dist):
        matrix = np.empty((self.paths_num, new_bboxes.shape[0]))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):                
                matrix[i, j] = self.tracks[i].get_position_distance(
                    new_bboxes[j],
                )
        return matrix

    def appearance_similarity_matrix(self, new_embeddings, max_dist):
        matrix = np.empty((self.paths_num, new_embeddings.shape[0]))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if len(self.tracks[i].history) == 0:
                    # track was not initialized
                    matrix[i, j] = max_dist
                else:
                    matrix[i, j] = self.tracks[i].get_similarity_distance(
                        new_embeddings[j])
        return matrix

    def run(self, boxes, embeddings, **kwargs):
        boxes = pre_process_boxes(boxes)
        boxes = np.array(boxes)
        if boxes.shape[0] == 0:
            return None

        boxes = Tracker.boxes_to_xywh(boxes)

        if self.tracks is None:
            if boxes.shape[0] < self.paths_num:
                self.tracks = [
                    self.track_class(boxes[i], embeddings[i], i + 1) for i in range(boxes.shape[0])
                ]
                for i in range(boxes.shape[0], self.paths_num):
                    self.tracks.append(self.track_class(track_id=i + 1))
            else:
                self.tracks = [
                    self.track_class(boxes[i], embeddings[i], i + 1) for i in range(self.paths_num)
                ]

        if self.track_class == KalmanTrack:
            for track in self.tracks:
                track.predict()

        t_1 = self.distance_cutoff
        # t_2 = 0.3
        t_2 = 10
        position_similarity_matrix = self.position_similarity_matrix(
            boxes, t_1)
        position_similarity_matrix[position_similarity_matrix > t_1] = t_1

        if self.steps < 10:
            appearance_similarity_matrix = np.zeros_like(
                position_similarity_matrix)
        else:
            appearance_similarity_matrix = self.appearance_similarity_matrix(
                embeddings, t_2)
            appearance_similarity_matrix[appearance_similarity_matrix > t_2] = t_2

        self.steps += 1

        admissible = np.logical_and(
            position_similarity_matrix < t_1, appearance_similarity_matrix < t_2
        )
        similarity_matrix = (
            (1 - self.appearance_weight) * position_similarity_matrix
            + self.appearance_weight * appearance_similarity_matrix
        )

        rows, cols = linear_sum_assignment(similarity_matrix)

        for i, j in zip(rows, cols):
            if not admissible[i, j] and len(self.tracks[i].history) > 0:
                # continue if track is initialized and not admissible
                continue
            # self.tracks[i].embedding = embeddings[j]
            self.tracks[i].embeddings.append(embeddings[j])
            # self.tracks[i].set_bbox(boxes[j])
            self.tracks[i].update(boxes[j])
