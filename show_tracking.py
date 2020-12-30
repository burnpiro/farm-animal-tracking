import time
from helpers.bb_helper import get_bb
from absl.flags import FLAGS
from absl import app, flags
from model.siamese.model import create_model
from model.siamese.config import cfg
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
import itertools
import numpy as np
import cv2
import os

import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
tf.config.set_visible_devices([], 'GPU')


flags.DEFINE_string('weights', f'{cfg.MODEL.WEIGHTS_PATH}siam-model-91_0.0518_0.5930.h5',
                    'path to weights file')
flags.DEFINE_integer('num', '7',
                     'number of objects to track')
flags.DEFINE_string('video', '.',
                    'path to video')


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
        pipeline_config = 'model/inference_graph/pipeline.config'
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

        siamese_net = create_model()
        siamese_net.load_weights(FLAGS.weights)

        cap = cv2.VideoCapture(FLAGS.video)

        i = 0
        frame_start = 0

        while(cap.isOpened()):
            frame_end = time.time()
            print(f'fps: {1/(frame_end-frame_start)}')
            frame_start = time.time()
            ret, frame = cap.read()
            i += 1

            frame = frame.astype('uint8')

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = tf.convert_to_tensor(
                np.expand_dims(rgb_frame, 0), dtype=tf.float32)

            detections, predictions_dict, shapes = detect_fn(input_tensor)

            boxes = detections['detection_boxes'][0]
            scores = detections['detection_scores'][0]

            resized = cv2.resize(frame, (800, 600))

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
            for idx, (box, _) in enumerate(boxes.items()):
                ymin, xmin, ymax, xmax = box
                bb_image = tf.image.crop_to_bounding_box(
                    input_tensor[0],
                    int(ymin*input_tensor[0].shape[0]),
                    int(xmin*input_tensor[0].shape[1]),
                    int((ymax - ymin)*input_tensor[0].shape[0]),
                    int((xmax - xmin)*input_tensor[0].shape[1])
                )
                bb_image = tf.image.resize(bb_image, size=(
                    cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE))
                boxes_tensors.append(bb_image)

            boxes_tensors = tf.stack(boxes_tensors)
            siamese_predictions = siamese_net(boxes_tensors).numpy()

            # set initial predictions
            boxes_list = list(boxes.items())
            if tracked_objects is None:
                tracked_objects = []
                for i in range(FLAGS.num):
                    (ymin, xmin, ymax, xmax), _ = boxes_list[i]
                    tracked_objects.append({'embedding': siamese_predictions[i], 'position': [
                        (ymin+ymax)/2, (xmin+xmax)/2]})

            else:
                previous_embeddings = np.array(
                    [x['embedding'] for x in tracked_objects])
                dot_prod = siamese_predictions @ previous_embeddings.transpose()
                distances = np.sum(siamese_predictions**2, axis=1)[..., np.newaxis] - \
                    2.0*dot_prod + np.sum(previous_embeddings**2, axis=1)
                distances[distances < 0] = 0.0
                mask = distances == 0.0
                distances = np.sqrt(distances)
                distances[mask] = 0

                # previous_positions = np.array(
                #     [x['position'] for x in tracked_objects])
                # positions = np.array([[(ymin+ymax)/2, (xmin+xmax)/2]
                #                       for (ymin, xmin, ymax, xmax), _ in boxes_list])

                # dot_prod = positions @ previous_positions.transpose()
                # distances = np.sum(positions**2, axis=1)[..., np.newaxis] - \
                #     2.0*dot_prod + np.sum(previous_positions**2, axis=1)[np.newaxis, ...]
                # distances[distances < 0] = 0.0
                # mask = distances == 0.0
                # distances = np.sqrt(distances)
                # distances[mask] = 0


                best_combination = None
                best_combination_dist = float('inf')
                if siamese_predictions.shape[0] >= FLAGS.num:
                    combinations = itertools.permutations(
                        range(siamese_predictions.shape[0]), FLAGS.num)

                    for combination in combinations:
                        combination = np.array(combination, dtype=np.int)
                        
                        dist = np.sum(distances[combination, np.arange(FLAGS.num)])
                        if dist < best_combination_dist:
                            best_combination_dist = dist
                            best_combination = combination

                    for i in range(len(tracked_objects)):
                        j = best_combination[i]
                        (ymin, xmin, ymax, xmax), _ = boxes_list[j]
                        tracked_objects[i] = {'embedding': siamese_predictions[j], 'position': [
                            (ymin+ymax)/2, (xmin+xmax)/2]}

                    # print(previous_positions)
                    # print(positions)
                    # print(distances)
                    # # print(best_combination)
                    # exit()

            for i, object in enumerate(tracked_objects):
                y, x = object['position']

                y = int(y*resized.shape[0])
                x = int(x*resized.shape[1])

                cv2.putText(resized, str(i), (x, y), font, .5,
                            (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('frame', resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
