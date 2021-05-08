import os.path
from os import path
import shutil
from shutil import copy2
import random
import json

random.seed(10)


path = "ndds_3dbox_data"
count = 1
for dirpath, dirnames, filenames in os.walk(path):

    # if os.path.isdir('{}/output'.format(dirpath)):
    #     shutil.rmtree('{}/output'.format(dirpath))
    # os.makedirs('{}/output'.format(dirpath))

    rel_dir = os.path.relpath(dirpath, path)
    for a in os.listdir(dirpath):
        try:
            b = os.listdir("{}/{}".format(path,a))
            b.sort()
        except:
            continue
    for file in b:
        rel_file = os.path.join(path, rel_dir, file)
        print(rel_file)
        if rel_file[-4:] == "json" and rel_dir != ".":
            print(rel_file)
            try:
                with open(file=rel_file) as f:
                    data = json.load(f)
                # with open(file="output/{}")
                for point in data['objects'][0]['projected_cuboid']:
                    print(point)
                print("==================")
            except:
                continue
            # number = random.random()
            # if number <= 0.9:
                with open(file="train_3dbox.txt", mode="a") as out_file:
                    print(rel_file)
                    out_file.write(rel_file)
                    out_file.write("\n")
            # else:
            #     with open(file="test_3dbox.txt", mode="a") as out_file:
            #         print(rel_file)
            #         out_file.write(rel_file)
            #         out_file.write("\n")



