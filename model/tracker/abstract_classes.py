from abc import ABC, abstractmethod


class AbstractTrack(ABC):
    def __init__(self) -> None:
        self.tracks = None

    @abstractmethod
    def initialize_track(self, track_id=None, **kwargs) -> None:
        pass

    @abstractmethod
    def update(self, bbox, embedding, **kwargs) -> None:
        """
         Updates current state of the track base on given bbox and embedding
        Args:
            bbox: List<y1,x1,y2,x2> - bbox coordinates top-left and bottom-right
            embedding: ndarrray - obj embedding

        Returns: None

        """
        pass

    @abstractmethod
    def update_with_prev_value(self) -> None:
        """
        Adds previous value to history in case some frames have to be skipped
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
    def skip_empty_frame(self) -> None:
        """
        Adds empty track value (updates with previous value) for every track
        used when there is a corrupted frame
        Returns: None
        """
        pass

    @abstractmethod
    def run(self, boxes, embeddings, **kwargs) -> None:
        """
        Updates all tracking objects (or add new ones) base on new bboxes and embeddings definition
        Runs every frame
        Args:
            boxes: Dict<bbox, classname>
            embeddings: List<Tensor>

        Returns: None

        """
        pass
