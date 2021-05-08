import glob
import os
import numpy as np
import sys

current_dir = "./new_DB/test/JPEGImages"
split_pct = 10  # 10% validation set
file_train = open("test.txt", "w")
file_val = open("test1.txt", "w")
counter = 1
index_test = round(100 / split_pct)
for fullpath in glob.iglob(os.path.join(current_dir, "*.png")):
  title, ext = os.path.splitext(os.path.basename(fullpath))
  if counter == index_test:
    counter = 1
    file_train.write(current_dir + "/" + title + '.png' + "\n")
  else:
    file_train.write(current_dir + "/" + title + '.png' + "\n")
    counter = counter + 1
file_train.close()
file_val.close()
