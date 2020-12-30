from model import model as Model


def model_generator(detection_model, prediction_model, tracking_method):
    return Model(detection_model, prediction_model, tracking_method)
