import glob
import os
import shutil

base_dir = ""
image_dir= ""

files = glob.glob(f"{base_dir}/temp_txt/*")

for file in files:
    with open(file, "r") as f:
        if f.read() == "":
            filename = file.split("/")[-1]
            filename_wo_ext = filename.split(".")[0]

            print(filename)
            os.remove(file)
            image_path = f"{image_dir}/{filename_wo_ext}.jpg"
            if os.path.exists(image_path):
                os.remove(f"{image_dir}/{filename_wo_ext}.jpg")
    