import numpy as np
from math import sqrt
from scipy.spatial.distance import euclidean
from model.tracker.abstract_classes import AbstractTrack

from helpers.bb_helper import bbox_to_position


class DefaultTrack(AbstractTrack):
    def __init__(self, track_id=None) -> None:
        super().__init__()
        self.track_id = None
        self.history = None

        self.initialize_track(track_id)

    def initialize_track(self, track_id=None, **kwargs) -> None:
        self.track_id = track_id
        self.history = []

    def update(self, bbox, embedding, **kwargs):
        self.history.append(bbox)

    def update_with_prev_value(self):
        self.update(self.history[-1], None)

    def has_history(self):
        """
        Checks if current track has any history
        Returns: boolean

        """
        return len(self.history) > 0

    def get_distance_to_box(self, bbox):
        """
        Calculates distance between last position in the history and given bbox
        If there is no last position available it returns max distance between 2 point

        Position is calculated based on center of the bbox
        Args:
            bbox: List<y1,x1,y2,x2>

        Returns: float
            Distance between two boxes or sqrt(2)

        """
        if not self.has_history():
            return sqrt(2)  # The max distance between any points in 2D space (1x1)

        last_position = self.history[-1]
        return euclidean(bbox_to_position(last_position), bbox_to_position(bbox))

    def get_history(self):
        return list(
            map(
                lambda x: bbox_to_position(x),
                self.history,
            )
        )
