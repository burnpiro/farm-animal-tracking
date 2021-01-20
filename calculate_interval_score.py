import os
import json
import codecs
import numpy as np


path = os.path.join('experiments/2021-01-19_23-29-03', "scores.json")
print(path)
scores = json.load(
    codecs.open(path)
)

for class_id, class_score in scores.items():
    mean_err = np.mean(class_score["intervals"]["parts"])
    scores[class_id]["intervals"]["mae"] = mean_err


json.dump(
    scores,
    codecs.open(path, "w", encoding="utf-8"),
    indent=2,
    sort_keys=False,
    separators=(",", ":"),
)
