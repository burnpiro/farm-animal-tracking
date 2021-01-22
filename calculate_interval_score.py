import os
import json
import codecs
import numpy as np
from data.names import names


path = os.path.join('experiments/2021-01-20_10-14-03', "scores.json")
print(path)
scores = json.load(
    codecs.open(path)
)

mapped_scores = []
full_values = []
for class_id, class_score in scores.items():
    mean_err = np.mean(class_score["intervals"]["parts"])
    print(class_id)
    print(names[int(class_id)-1], mean_err)
    mapped_scores.append((names[int(class_id)-1], mean_err))
    full_values.append((names[int(class_id)-1], class_score["total"]["mae"]))
    scores[class_id]["intervals"]["mae"] = mean_err

print()

sorted_by = sorted(mapped_scores, key=lambda tup: tup[1])
for class_id, class_score in sorted_by:
    print(class_id, "{:.2f}".format(class_score*100))

print()

sorted_by = sorted(full_values, key=lambda tup: tup[1])
for class_id, class_score in sorted_by:
    print(class_id, "{:.2f}".format(class_score*100))


json.dump(
    scores,
    codecs.open(path, "w", encoding="utf-8"),
    indent=2,
    sort_keys=False,
    separators=(",", ":"),
)
