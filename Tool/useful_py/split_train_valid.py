import glob
import os
import shutil

base_dir = ""

train_files = glob.glob(f"{base_dir}/train/*")

def split_files():

    image_files = [img_file for img_file in train_files if img_file.split(".")[-1] == "jpg"]
    label_files = [label_file for label_file in train_files if label_file.split(".")[-1] == "txt"]

    file_len = len(label_files)
    split_num = int(file_len * 0.2)

    valid_label_files = label_files[:split_num]

    for file in valid_label_files:
        filename = file.split("/")[-1]
        filename_wo_ext = filename.split(".")[0]
        img_filename = f"{filename_wo_ext}.jpg"

        shutil.move(file, f"{base_dir}/valid/{filename}")

        if os.path.exists(f"{base_dir}/train/{img_filename}"):
            shutil.move(f"{base_dir}/train/{img_filename}", f"{base_dir}/valid/{img_filename}")


if __name__ == "__main__":
    split_files()    



