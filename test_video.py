import sys
import cv2
from tqdm import tqdm

if len(sys.argv) != 2:
    print(f"USAGE: {sys.argv[0]} <path_to_video>")
    exit()

from model.detection_model.detection_model import DefaultDetectionModel
from model.siamese.siamese_model import DefaultSiameseModel
from model.tracker.tracker import Tracker
from model.model import Model

model = Model(DefaultDetectionModel(), DefaultSiameseModel(), Tracker(7))

paths = model.predict_video(sys.argv[1])

print(paths)

cap = cv2.VideoCapture(sys.argv[1])

i = 0
frame_start = 0
num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total=num_of_frames)

width = 2688
height = 1520
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("out.mp4", fourcc, 5, (2688, 1520))

while cap.isOpened():
    ret, frame = cap.read()
    if i > num_of_frames:
        break
    if frame is None:
        break

    frame = frame.astype("uint8")
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    i += 1
    pbar.update(i)
    for idx, track_obj in paths.items():
        if i > len(track_obj):
            continue
        ymin, xmin, ymax, xmax = track_obj[i - 1]
        rgb_frame = cv2.rectangle(
            rgb_frame,
            (int(xmin * width), int(ymin * height)),
            (int(xmax * width), int(ymax * height)),
            (0, 255, 0),
            2,
        )

    cv2.imshow("frame", rgb_frame)
    out.write(rgb_frame)

out.release()
cv2.destroyAllWindows()
