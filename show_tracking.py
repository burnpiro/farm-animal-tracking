import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
# tf.config.set_visible_devices([], 'GPU')

import time
from helpers.bb_helper import get_bb
from absl.flags import FLAGS
from absl import app, flags
from model.siamese.classification_model import create_model, create_embedding_model
from model.siamese.config import cfg
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
import itertools
import numpy as np
import cv2
import os

from model.tracker.tracker import Tracker
from model.siamese.siamese_model import DefaultSiameseModel
from model.tracker import get_embeddings

# flags.DEFINE_string('weights', f'{cfg.MODEL.WEIGHTS_PATH}siam-model-91_0.0518_0.5930.h5',
#                     'path to weights file')
flags.DEFINE_integer('num', '16',
                     'number of objects to track')
flags.DEFINE_string('video', '.',
                    'path to video')

WINDOW_SIZE = (800, 600)

def get_keypoint_tuples(eval_config):
    """Return a tuple list of keypoint edges from the eval config.

    Args:
      eval_config: an eval config containing the keypoint edges

    Returns:
      a list of edge tuples, each in the format (start, end)
    """
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
        tuple_list.append((edge.start, edge.end))
    return tuple_list


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


def main(argv):
    tracked_objects = None
    font = cv2.FONT_HERSHEY_SIMPLEX

    if __name__ == '__main__':
        pipeline_config = 'model/detection_model/inference_graph/pipeline.config'
        model_dir = 'model/detection_model/inference_graph/checkpoint'

        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        model_config = configs['model']
        detection_model = model_builder.build(
            model_config=model_config, is_training=False)

        # detection_model.feature_extractor.build((640, 640))

        # print(detection_model.feature_extractor.classification_backbone)
        # for layer in detection_model.feature_extractor.classification_backbone.layers:
        #     print(layer.name)

        # exit()

        ckpt = tf.compat.v2.train.Checkpoint(
            model=detection_model)
        ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

        detect_fn = get_model_detection_function(detection_model)

        label_map_path = os.path.join(
            "model/detection_model/", configs['eval_input_config'].label_map_path)
        label_map = label_map_util.load_labelmap(label_map_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=label_map_util.get_max_label_map_index(label_map),
            use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        label_map_dict = label_map_util.get_label_map_dict(
            label_map, use_display_name=True)

        # siamese_net = create_model()
        # siamese_net.load_weights(FLAGS.weights)

        # siamese_net = create_embedding_model(siamese_net)

        cap = cv2.VideoCapture(FLAGS.video)

        i = 0
        frame_start = 0

        # tracker = Tracker(paths_num=FLAGS.num, appearance_weight=0.5, max_euclidean_distance=10)
        tracker = Tracker(paths_num=FLAGS.num)
        weights_dir = os.path.join(
            "model/siamese/weights", "MobileNetV2", "siam-118-0.0001-1.0a_0.0633.h5"
        )
        base_model = 'MobileNetV2'
        siamese_obj = DefaultSiameseModel(weights_path=weights_dir, base_model=base_model)

        out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, WINDOW_SIZE)

        while(cap.isOpened()):
            frame_end = time.time()
            print(f'fps: {1/(frame_end-frame_start)}')
            frame_start = time.time()
            ret, frame = cap.read()
            i += 1

            if frame is None:
                break

            frame = frame.astype('uint8')

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = tf.convert_to_tensor(
                np.expand_dims(rgb_frame, 0), dtype=tf.float32)

            detections, predictions_dict, shapes = detect_fn(input_tensor)

            resized = cv2.resize(frame, WINDOW_SIZE)

            boxes = get_bb(
                resized,
                detections['detection_boxes'][0].numpy(),
                (detections['detection_classes'][0].numpy() + 1).astype(int),
                detections['detection_scores'][0].numpy(),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False,
                keypoints=None,
                keypoint_scores=None,
                keypoint_edges=get_keypoint_tuples(configs['eval_config']))


            boxes_tensors = []
            for box in boxes:
                x1, y1, x2, y2 = box
                width = x2-x1
                height = y2-y1

                [bb_image] = tf.image.crop_to_bounding_box(
                    input_tensor,
                    int(y1*input_tensor.shape[1]),
                    int(x1*input_tensor.shape[2]),
                    int(height*input_tensor.shape[1]),
                    int(width*input_tensor.shape[2])
                )

                boxes_tensors.append(bb_image)

            embeddings = siamese_obj.predict(boxes_tensors)
            # embeddings = get_embeddings(input_tensor, list(boxes.keys()), siamese_net, cfg.NN.INPUT_SIZE)
            tracker.run(boxes, embeddings)
            history = tracker.get_history()
            
            for track_id, track_history in history.items():
                x, y = track_history[-1]

                x = int(x*resized.shape[1])
                y = int(y*resized.shape[0])                

                cv2.putText(resized, str(track_id), (x, y), font, .5,
                            (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('frame', resized)
            # if out:
            out.write(resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        # if out:
        #     out.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
