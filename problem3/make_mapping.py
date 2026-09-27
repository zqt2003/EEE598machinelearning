"""Create LOC_synset_mapping.txt ("n01440764 tench") for the ImageNet class folders on Sol.

Uses the standard ImageNet class index (index -> [synset ID, name]) and looks up each folder by
its synset ID, so it works even if the folder has only some of the 1000 classes.
"""
import json
import os
import urllib.request

ROOT = "/data/datasets/community/deeplearning/imagenet/train"
URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
LOCAL = "imagenet_class_index.json"

if not os.path.exists(LOCAL):
    urllib.request.urlretrieve(URL, LOCAL)
index = json.load(open(LOCAL))
names = {wnid: name.replace("_", " ") for wnid, name in index.values()}     # 1000 standard classes

wnids = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
missing = [w for w in wnids if w not in names]
with open("LOC_synset_mapping.txt", "w") as f:
    for w in wnids:
        f.write(f"{w} {names.get(w, w)}\n")

print(f"{len(wnids)} class folders found, {len(wnids) - len(missing)} matched a name")
if missing:
    print(f"no name for {len(missing)} folders (their ID is used as the name), e.g. {missing[:5]}")
for w in wnids[:5]:
    print(" ", w, names.get(w, "?"))
print("wrote LOC_synset_mapping.txt")
