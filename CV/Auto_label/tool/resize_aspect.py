import cv2
import numpy as np
import glob
import os
import argparse


def resize_image_label(img_files, seg_files, output_image_dir, output_label_dir, resize_size):
    for i, (img, seg) in enumerate(zip(img_files, seg_files)):
        image = cv2.imread(img)
        basename = os.path.basename(img)

        image_width = image.shape[1]
        image_height = image.shape[0]

        if image_width > image_height:
            new_width = resize_size
            new_height = int(image_height * (resize_size / image_width))
        else:
            new_width = int(image_width * (resize_size / image_height))
            new_height = resize_size

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        image_width = resized_image.shape[1]
        image_height = resized_image.shape[0]

        with open(seg, "r") as f:
            segmentation_data_list = f.readlines()

            resized_segmentation_data_list = []

            for segmentation_data in segmentation_data_list:
                segmentation_data = segmentation_data.split()
                if len(segmentation_data) == 0:
                    break

                # class_id = int(segmentation_data[0])
                class_id = 0

                points = [float(point) for point in segmentation_data[1:]]
                num_points = len(points) // 2
                points = [(points[i * 2], points[i * 2 + 1]) for i in range(num_points)]

                resized_points = []
                for point in points:
                    resized_x = float(point[0] * new_width / image_width)
                    resized_y = float(point[1] * new_height / image_height)
                    if resized_x < 0:
                        resized_x = 0
                    elif resized_x >= new_width:
                        resized_x = new_width - 1
                    if resized_y < 0:
                        resized_y = 0
                    elif resized_y >= new_height:
                        resized_y = new_height - 1

                    resized_points.append((resized_x, resized_y))

                new_segmentation_data = f"{class_id}"
                for point in resized_points:
                    new_segmentation_data += f" {point[0]} {point[1]}"
                new_segmentation_data += "\n"
                resized_segmentation_data_list.append(new_segmentation_data)

            output_image_file = os.path.join(output_image_dir, f"{i}_" + os.path.basename(img))
            cv2.imwrite(output_image_file, resized_image)

            output_seg_file = os.path.join(output_label_dir, f"{i}_" + os.path.splitext(basename)[0] + ".txt")
            with open(output_seg_file, "w") as f:
                f.writelines(resized_segmentation_data_list)


def drew_ploygon(img_file, seg_file, output_dir):

    for i,(img,seg) in enumerate(zip(img_file,seg_file)):
        image = cv2.imread(img)
        basename = os.path.basename(img)

        image_width = image.shape[1]
        image_height = image.shape[0]

        with open(seg, "r") as f:
            # segmentation_data = f.read().strip().split()
            segmentation_data_list = f.readlines()
            # print("segmentation_data_list",segmentation_data_list)
            
            for segmentation_data in segmentation_data_list:
                segmentation_data = segmentation_data.split()
                if len(segmentation_data) == 0:
                    break
                class_id = int(segmentation_data[0])
                print("class_id",class_id)
                points = [float(point) for point in segmentation_data[1:]]
                num_points = len(points) // 2
                points = [(int(points[i * 2] * image_width), int(points[i * 2 + 1] * image_height)) for i in range(num_points)]

                points = np.array(points)
                # print("points",points)
                cv2.polylines(image, [points], True, (0, 255, 0), thickness=2)
                
        cv2.imwrite(output_dir + f"/{i}_" + basename, image)


if __name__ == "__main__":

    image_path = "./images"
    label_path = "./labels"
    output_image_dir = "./result_image"
    output_label_dir = "./result_label"
    
    # resize_size = size
    resize_size = 960
    # on or off
    set_drew = "on"

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    img_file = sorted(glob.glob(image_path + "/*jpg"))
    seg_file = sorted(glob.glob(label_path + "/*txt"))


    resize_image_label(img_file, seg_file, output_image_dir, output_label_dir, resize_size)

    if set_drew == "on":
        image_path = output_image_dir
        label_path = output_label_dir
        output_dir = "./drew_result"

        os.makedirs(output_dir, exist_ok=True)

        img_file = sorted(glob.glob(image_path + "/*jpg"))
        seg_file =  sorted(glob.glob(label_path + "/*txt"))

        drew_ploygon(img_file, seg_file, output_dir)
