import numpy as np
from model.tracker.abstract_classes import AbstractTrack


class DefaultTrack(AbstractTrack):
    def __init__(self, track_id=None) -> None:
        super().__init__()
        self.track_id = None
        self.history = None

        self.initialize_track(track_id)

    def initialize_track(self, track_id=None) -> None:
        self.track_id = track_id
        self.history = []

    def update(self, bbox):
        self.history.append(bbox)

    def get_history(self):
        return self.history
