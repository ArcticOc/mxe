import os
import shutil

image_mapping = {}
with open("data/splits/cub_splits/cub_images.txt", "r") as file:
    for line in file:
        parts = line.strip().split()
        image_mapping[int(parts[0])] = parts[1]

split_mapping = {}
with open("data/splits/cub_splits/cub_split.txt", "r") as file:
    for line in file:
        parts = line.strip().split()
        split_mapping[int(parts[0])] = int(parts[1])

train_path = "data/CUB_200_2011/train"
test_path = "data/CUB_200_2011/test"

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)


for image_id, image_path in image_mapping.items():
    source_path = os.path.join("data/CUB_200_2011/images", image_path)
    target_folder = train_path if split_mapping.get(image_id, 1) == 1 else test_path
    target_path = os.path.join(target_folder, image_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.move(source_path, target_path)
