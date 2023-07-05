import glob
import os
import shutil

img_dir = "image path"
label_dir = "label path"

images = glob.glob(f"{img_dir}/*")
labels = glob.glob(f"{label_dir}/*")
print(len(images))
print(len(labels))

for image in images:
    filename = image.split("/")[-1].split(".")[0]

    if not os.path.exists(f"{label_dir}/{filename}.txt"):
        os.remove(image)
print(len(image))

for file in labels:
    filename = file.split("/")[-1].split(".")[0]

    if not os.path.exists(f"{img_dir}/{filename}.jpg"):
        os.remove(file)
print(len(file))