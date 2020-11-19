import glob
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


class DataGenerator:
    def __init__(self, folder_path="./data/cropped_animals", file_ext="jpg", debug=False):
        """
        Args:
            folder_path: string ## Path to folder with video frames
            file_ext: string (optional) looking for files with this extension
            debug: boolean (optional) should generator display any warnings?
        """
        self.images = []
        self.debug = debug
        self.data_path = folder_path

        if not os.path.isdir(folder_path):
            print("Images folder path {} does not exist. Exiting...".format(folder_path))
            sys.exit()

        for video_dir in os.scandir(folder_path):
            for class_dir in os.scandir(video_dir.path):
                for file in glob.glob(f"{class_dir.path}/*.{file_ext}"):
                    self.images.append((file, class_dir.name))
            break

        self.images = pd.DataFrame(self.images, columns=['path', 'label'])
        print(f'Found {len(self.images)} files for {len(self.images["label"].unique())} unique classes')

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
        target = self.images.pop('label').progress_map(DataGenerator.process_label).to_numpy()
        images = self.images.pop('path').progress_map(DataGenerator.process_image).to_numpy()
        reshaped_images = np.concatenate(images).reshape(
            (images.shape[0], images[1].shape[0], images[1].shape[1], images[1].shape[2]))
        ds = tf.data.Dataset.from_tensor_slices((reshaped_images, target))
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(cfg.TRAIN.BATCH_SIZE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
