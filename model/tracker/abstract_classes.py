from abc import ABC, abstractmethod


class AbstractTrack(ABC):
    def __init__(self) -> None:
        self.tracks = None

    @abstractmethod
    def initialize_track(self, track_id=None) -> None:
        pass

    @abstractmethod
    def update(self, bbox) -> None:
        pass

    @abstractmethod
    def get_history(self) -> list:
        pass


class AbstractTracker(ABC):
    def __init__(self) -> None:
        self.tracks = None

        self.initialize_tracker()

    @abstractmethod
    def initialize_tracker(self) -> None:
        pass

    @abstractmethod
    def get_history(self) -> dict:
        pass

    @abstractmethod
    def run(self, boxes, embeddings) -> None:
        pass
