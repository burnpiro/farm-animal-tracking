import os
import json
import codecs
from pathlib import Path
import numpy as np
from data.names import names

# experiments/tracking/MobileNetV2/AvgEmbeddingTracker/11_nursery_high_activity_day-cropped.mp4_2021-01-24_13-26-02/scores.json
results_total = np.empty((17,3), dtype="U5")
results_interval = np.empty((17,3), dtype="U5")
# print(results)

i = 0
for path in Path('experiments/tracking').rglob('15_nur*/scores.json'):
    print(path)
    scores = json.load(
        codecs.open(path)
    )
    # continue

    mapped_scores = []
    full_values = []
    for class_id, class_score in scores.items():
        mean_err = np.mean(class_score["intervals"]["parts"])
        print(class_id)
        print(names[int(class_id)-1], mean_err)
        mapped_scores.append((names[int(class_id)-1], mean_err))
        full_values.append((names[int(class_id)-1], class_score["total"]["avg_err"]))
        scores[class_id]["intervals"]["avg_err"] = mean_err

    print()

    sorted_by = sorted(mapped_scores, key=lambda tup: tup[0])
    j = 0
    avg_total = 0.0
    for class_id, class_score in sorted_by:
        results_total[j][i] = "{:.2f}".format(class_score)
        avg_total += class_score
        j += 1
        # print(class_id, "{:.2f}".format(class_score))

    results_total[16][i] = "{:.2f}".format(avg_total/16)
    print()

    sorted_by = sorted(full_values, key=lambda tup: tup[0])
    j = 0
    avg_total = 0.0
    for class_id, class_score in sorted_by:
        results_interval[j][i] = "{:.2f}".format(class_score)
        avg_total += class_score
        j += 1
        print(class_id, "{:.2f}".format(class_score))
    results_interval[16][i] = "{:.2f}".format(avg_total/16)


    json.dump(
        scores,
        codecs.open(path, "w", encoding="utf-8"),
        indent=2,
        sort_keys=False,
        separators=(",", ":"),
    )
    i += 1

print(results_total)

print(results_interval)