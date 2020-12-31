from abc import ABC, abstractmethod


class AbstractModel(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict_image(self, path_to_img: str):
        pass

    @abstractmethod
    def predict_video(self, path_to_video: str):
        pass

    @abstractmethod
    def print_bb_on_image(self, path_to_img: str):
        pass

    @abstractmethod
    def recognize_animals_on_image(self, path_to_img: str):
        pass
