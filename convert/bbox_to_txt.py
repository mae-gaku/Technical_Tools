from PIL import Image
import os
import pathlib
import glob
import cv2


work_dir = ""
image_name = "images"
label_name = "labels"
image_list = glob.glob(work_dir+"/" + image_name + "/*")
label_list =  glob.glob(work_dir+"/" + label_name +  "/*")

for img,labe in zip(image_list,label_list):
    image = cv2.imread(img)                
    height, width = image.shape[0:2]

    # Get a list of all the classes used in the yolo format
    with open(labe) as f:
        yolo_classes = f.read().split(' ')
    array_of_yolo_classes = []
    for x in yolo_classes:
        array_of_yolo_classes.append(x.strip())

    if len(array_of_yolo_classes) < 6:
        print(array_of_yolo_classes)
    else:
            
        print(array_of_yolo_classes[1])
        xmin = float(array_of_yolo_classes[1])
        ymin = float(array_of_yolo_classes[2])
        xmax = float(array_of_yolo_classes[3])
        ymax = float(array_of_yolo_classes[4])

        el_1 = (xmin + xmax) / 2 / width
        el_2 = (ymin + ymax) / 2 / height
        el_3 = (xmax - xmin) / width
        el_4 = (ymax - ymin) / height


        text = f"0 {el_1:.10f} {el_2:.10f} {el_3:.10f} {el_4:.10f}\n"
        print(text)
        base_dir = ""

        new_filename = os.path.basename(labe)
        if not os.path.exists(f"{base_dir}/output/{new_filename}"):
            with open(f"{base_dir}/output/{new_filename}", "w", newline="\n") as f:
                f.write(text)

        else:
            with open(f"{base_dir}/output/{new_filename}", "a", newline="\n") as f:
                f.write(text)

