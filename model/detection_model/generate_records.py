import tensorflow as tf
from model.detection_model.object_detection.utils import dataset_util

import os
import json

TRAIN_OUT = 'train.record'
TEST_OUT = 'test.record'
VALIDATION_SPLIT = 0.3

def create_tf_example(example):
    # TODO(user): Populate the following variables from your example.
    height = 640  # Image height
    width = 640  # Image width
    filename = example['path']  # Filename of the image. Empty if image is not from file
    with open(filename, 'rb') as f:
        encoded_image_data = f.read()  # Encoded image bytes
    image_format = b'jpeg'  # b'jpeg' or b'png'

    # List of normalized left x coordinates in bounding box (1 per box)
    xmins = [bbox[0] for bbox in example['bboxes']]
    xmaxs = [bbox[2] for bbox in example['bboxes']]  # List of normalized right x coordinates in bounding box
    # (1 per box)
    # List of normalized top y coordinates in bounding box (1 per box)
    ymins = [bbox[1] for bbox in example['bboxes']]
    ymaxs = [bbox[3] for bbox in example['bboxes']] # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = ['pig'.encode('utf-8') for bbox in example['bboxes']]  # List of string class name of bounding box (1 per box)
    classes = [1 for bbox in example['bboxes']]  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main():
    train_writer = tf.io.TFRecordWriter(TRAIN_OUT)
    test_writer = tf.io.TFRecordWriter(TEST_OUT)

    examples = []

    for entry in os.scandir('frames'):
        annotations_path = os.path.join(entry, 'annotations.json')
        anntations = None
        with open(annotations_path, 'r') as f:
            anntations: dict = json.loads(f.read())

        data_len = len(anntations)
        test_size = int(data_len*VALIDATION_SPLIT)
        train_size = data_len-test_size
        
        for i, (k, v) in enumerate(anntations.items()):
            example = {
                "path" : k,
                "bboxes" : v,
                "set": "train" if i < train_size else "test"
            }

            examples.append(example)


        

    for example in examples:
        tf_example = create_tf_example(example)
        if example['set'] == 'train':
            train_writer.write(tf_example.SerializeToString())
        elif example['set'] == 'test':
            test_writer.write(tf_example.SerializeToString())
        else:
            raise Exception(f'Unknown dataset: { example["set"] }')

    train_writer.close()
    test_writer.close()


if __name__ == '__main__':
    main()
