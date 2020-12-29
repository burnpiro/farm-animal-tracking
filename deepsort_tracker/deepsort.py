from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.application_util import preprocessing as prep
from deep_sort.application_util import visualization
from deep_sort.deep_sort.detection import Detection

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from siamese.config import cfg

from scipy.stats import multivariate_normal

# def get_gaussian_mask():
# 	#128 is image size
# 	x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
# 	xy = np.column_stack([x.flat, y.flat])
# 	mu = np.array([0.5,0.5])
# 	sigma = np.array([0.22,0.22])
# 	covariance = np.diag(sigma**2)
# 	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
# 	z = z.reshape(x.shape)

# 	z = z / z.max()
# 	z  = z.astype(np.float32)

# 	mask = torch.from_numpy(z)

# 	return mask


class deepsort_rbc():
    def __init__(self, model):
        self.encoder = model

        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", .5, 100)
        self.tracker = Tracker(self.metric)

        # self.gaussian_mask = get_gaussian_mask().cuda()


    def reset_tracker(self):
        self.tracker = Tracker(self.metric)


    def pre_process(self, frame, detections):
        boxes_tensors = []
        for box in detections:
            ymin, xmin, ymax, xmax = box
            # print(frame.shape)
            # print(box)
            # print(int(ymin*frame.shape[1]),
            #       int(xmin*frame.shape[2]),
            #       int((ymax - ymin)*frame.shape[1]),
            #       int((xmax - xmin)*frame.shape[2]))
            bb_image = tf.image.crop_to_bounding_box(
                frame,
                int(ymin*frame.shape[1]),
                int(xmin*frame.shape[2]),
                int((ymax - ymin)*frame.shape[1]),
                int((xmax - xmin)*frame.shape[2])
            )
            bb_image = tf.image.resize(bb_image, size=(
                cfg.NN.INPUT_SIZE, cfg.NN.INPUT_SIZE))
            boxes_tensors.append(bb_image)

        boxes_tensors = tf.concat(boxes_tensors, axis=0)

        return boxes_tensors


    def run_deep_sort(self, frame, out_scores, out_boxes):

        if out_boxes == []:
            self.tracker.predict()
            print('No detections')
            trackers = self.tracker.tracks
            return trackers

        detections = np.array(out_boxes)
        # print(detections)
        #features = self.encoder(frame, detections.copy())

        processed_crops = self.pre_process(frame, detections)
        detections[:, 2] -= detections[:, 0]
        detections[:, 3] -= detections[:, 1]

        # print(detections)
        # processed_crops = self.gaussian_mask * processed_crops

        features = self.encoder.predict(processed_crops)

        dets = [Detection(bbox, score, feature)
                for bbox, score, feature in
                zip(detections, out_scores, features)]

        outboxes = np.array([d.tlwh for d in dets])

        outscores = np.array([d.confidence for d in dets])
        indices = prep.non_max_suppression(outboxes, 0.8, outscores)

        dets = [dets[i] for i in indices]

        self.tracker.predict()
        self.tracker.update(dets)

        return self.tracker, dets
