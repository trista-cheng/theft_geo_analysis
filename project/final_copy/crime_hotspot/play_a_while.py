from classifier import transformer, find_spot

import json
import glob

# X_train, X_test, y_train, y_test = model.read_data("car", "classifier/mergedata")
l = json.loads('["台北市信義區松山路514號",25.0355015,121.5636883]')
# transformer.generate_features([l], 'car')
# l = glob.glob('./**/district.xlsx', recursive=True)
d = find_spot.near_point('car', l[1], l[2], num=2)[['label', 'lat']]
# d.insert(0, "label", d.index)
print(d)