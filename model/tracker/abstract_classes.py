from abc import ABC, abstractmethod


class AbstractTrack(ABC):
    def __init__(self) -> None:
        self.tracks = None

    @abstractmethod
    def initialize_track(self, track_id=None) -> None:
        pass

    @abstractmethod
    def update(self, bbox) -> None:
        """
         Updates current state of the track base on given bbox
        Args:
            bbox: List<y1,x1,y2,x2>

        Returns: None

        """
        pass

    @abstractmethod
    def get_history(self) -> list:
        """
        Parses Track history into list of positions over time
        Returns: List<List<x,y>>
            List of x,y positions from stored history
        """
        pass


class AbstractTracker(ABC):
    def __init__(self) -> None:
        self.tracks = None

    @abstractmethod
    def reset_tracker(self) -> None:
        pass

    @abstractmethod
    def get_history(self) -> dict:
        """
        Returns list of positions per every tracked object
        Returns:
            Dict<object_id, List<x,y>>
        """
        pass

    @abstractmethod
    def run(self, boxes, embeddings) -> None:
        """
        Updates all tracking objects (or add new ones) base on new bboxes and embeddings definition
        Runs every frame
        Args:
            boxes: Dict<bbox, classname>
            embeddings: List<Tensor>

        Returns: None

        """
        pass
