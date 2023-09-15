import os
import pathlib
import glob
import re
dataset_name = "txt"
work_dir = ""
# image_list = os.listdir(work_dir+"/"+dataset_name+"/images")

image_list = glob.glob(work_dir+"/"+dataset_name + "/*")

for file_name in image_list:
    print(file_name)
    with open(file_name, encoding="cp932") as f:
        data_lines = f.read()
    data_lines = re.sub('^', '0', data_lines)
    # data_lines = data_lines.replace("0", "1")

    with open(file_name, mode="w", encoding="cp932") as f:
        f.write(data_lines)