import random
from os import listdir
import os
import glob
import shutil
from shutil import copyfile
#
# for i in range(4):
#     print(random.randint(0, 310))

random.seed(1)

mini = 10

datasets = ["birds", "flowers", "scenes"]

for dataset in datasets:
    train_path = "dataset/" + dataset + "/train"
    class_idx = os.listdir(train_path)
    class_num = len(class_idx)
    os.mkdir("dataset/" + dataset + "/mini")
    for i in range(class_num):
        os.mkdir("dataset/" + dataset + "/mini/" + class_idx[i])

        class_path = train_path + "/" + class_idx[i]
        files = os.listdir(class_path)
        dir_path = "dataset/" + dataset + "/mini/" + class_idx[i]
        rand = random.sample(range(0, len(files)), mini)

        for j in rand:
            shutil.copy(train_path + "/" + class_idx[i] + "/" + files[j], dir_path)
