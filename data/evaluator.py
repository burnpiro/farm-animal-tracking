import glob
import math
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

tqdm.pandas()

"""
Files have to be stored in a structure:

tracking/
    01/
        frames_tracking.json
        pigs_tracking.json
    02/
        frames_tracking.json
        pigs_tracking.json
    ...
"""

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Evaluator:
    def __init__(
        self,
    ):
        """
        Args:
        """
        self.images = None
