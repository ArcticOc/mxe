import os
import random
import shutil

random.seed(2011)
source_dir = "data/CUB_200_2011"

train_dir = os.path.join(source_dir, "train")
val_dir = os.path.join(source_dir, "val")
test_dir = os.path.join(source_dir, "test")


os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


class Split:
    def __init__(self, source_dir, train_dir, val_dir, test_dir):
        self.source_dir = source_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

    def split_method_1(self):
        """popular method [train: 100 classes, val: 50 classes, test: 50 classes]"""
        class_folders = [
            f
            for f in os.listdir(self.source_dir)
            if os.path.isdir(os.path.join(self.source_dir, f))
        ]

        # Sort class folders to ensure correct ordering by index
        class_folders.sort(key=lambda x: int(x.split(".")[0]))

        for i, folder in enumerate(class_folders):
            source_path = os.path.join(self.source_dir, folder)
            if i % 2 == 0:  # Even index for train
                shutil.move(source_path, self.train_dir)
            elif i % 4 == 1:  # Index % 4 == 1 for val
                shutil.move(source_path, self.val_dir)
            elif i % 4 == 3:  # Index % 4 == 3 for test
                shutil.move(source_path, self.test_dir)

    def split_method_2(self):
        """Randon select [train: 128 classes, val: 40 classes, test: 32 classes]"""
        class_folders = [
            f
            for f in os.listdir(self.source_dir)
            if os.path.isdir(os.path.join(self.source_dir, f))
        ]
        class_folders.sort(key=lambda x: int(x.split(".")[0]))

        train_folders = random.sample(class_folders, 128)

        remaining_folders = [f for f in class_folders if f not in train_folders]
        val_folders = random.sample(remaining_folders, 40)

        test_folders = [f for f in remaining_folders if f not in val_folders]

        for folder in train_folders:
            shutil.move(os.path.join(self.source_dir, folder), self.train_dir)
        for folder in val_folders:
            shutil.move(os.path.join(self.source_dir, folder), self.val_dir)
        for folder in test_folders:
            shutil.move(os.path.join(self.source_dir, folder), self.test_dir)


split_method = Split(source_dir, train_dir, val_dir, test_dir)
split_method.split_method_2()
