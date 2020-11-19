import glob
import math
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

tqdm.pandas()

from siamese.config import cfg

"""
Files have to be stored in a structure:

main_folder/
    1/
        1/
            0030.jpg
            1080.jpg
            ...
        2/
            2400.jpg
            ...
    2/
        2/
            5230.jpg
            ...
        14/
            8800.jpg
            ...
            
This structure is going to extract images for 3 classes [1,2,14] from 2 videos [1,2]. 
Class present in more than one video will be combined into one (like class "2" in the example)
"""

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, folder_path="./data/cropped_animals", file_ext="jpg", debug=False, training=True):
        """
        Args:
            folder_path: string ## Path to folder with video frames
            file_ext: string (optional) looking for files with this extension
            debug: boolean (optional) should generator display any warnings?
        """
        self.images = None
        self.debug = debug
        self.data_path = folder_path

        if not os.path.isdir(folder_path):
            print("Images folder path {} does not exist. Exiting...".format(folder_path))
            sys.exit()

        images = []
        for video_dir in os.scandir(folder_path):
            for class_dir in os.scandir(video_dir.path):
                for i, file in enumerate(glob.glob(f"{class_dir.path}/*.{file_ext}")):
                    # For train set it to "i%5 == 1"
                    if training:
                        if i % 5 == 1:
                            continue
                    else:
                        if i % 5 != 1:
                            continue

                    images.append((file, class_dir.name))

        images = pd.DataFrame(images, columns=['path', 'label'])
        low_class_count = min(images["label"].value_counts())
        unique_classes = images["label"].unique()

        class_dfs = {}
        for class_id in unique_classes:
            class_dfs[str(class_id)] = images[images['label'] == class_id].sample(frac=1).reset_index(drop=True)

        batched = []
        for i in range(0, low_class_count - 1, 2):
            for class_id in unique_classes:
                rows = class_dfs[str(class_id)].loc[[i, i + 1], :]
                batched.append(rows.to_numpy())

        batched = np.array(batched)

        batched = batched.reshape((batched.shape[0] * batched.shape[1], batched.shape[2]))
        self.images = pd.DataFrame(batched, columns=['path', 'label'])
        print(f'Found {len(self.images)} files for {len(self.images["label"].unique())} unique classes')

    def __len__(self):
        return math.ceil(len(self.images) / cfg.TRAIN.BATCH_SIZE)

    @staticmethod
    def process_image(image_path):
        """
        Args:
            image_path: string

        Returns:
            ((cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE, 3), class)

        """
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image - tf.keras.applications.mobilenet_v2.preprocess_input(image)

        return image[0]

    @staticmethod
    def process_label(label):
        """
        Args:
            label: string

        Returns:
            int

        """
        return int(label)

    def get_dataset(self):
        """
        Returns:
            tf.Dataset
        """
        target = self.images.pop('label').progress_map(DataGenerator.process_label).to_numpy()
        images = self.images.pop('path').progress_map(DataGenerator.process_image).to_numpy()
        reshaped_images = np.concatenate(images).reshape(
            (images.shape[0], images[1].shape[0], images[1].shape[1], images[1].shape[2]))
        ds = tf.data.Dataset.from_tensor_slices((reshaped_images, target))
        ds = ds.cache()
        ds = ds.batch(cfg.TRAIN.BATCH_SIZE)
        ds = ds.prefetch(buffer_size=cfg.TRAIN.BATCH_SIZE)
        return ds

    def __getitem__(self, item):
        images = self.images.loc[item * cfg.TRAIN.BATCH_SIZE:(item + 1) * cfg.TRAIN.BATCH_SIZE]
        target = images.pop('label').map(DataGenerator.process_label).to_numpy()
        images = images.pop('path').map(DataGenerator.process_image).to_numpy()
        reshaped_images = np.concatenate(images).reshape(
            (images.shape[0], images[1].shape[0], images[1].shape[1], images[1].shape[2]))

        return reshaped_images, target
