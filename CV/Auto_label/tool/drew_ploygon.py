import cv2
import numpy as np
import glob
import os

image_path = ""
label_path = ""
output_dir = ""

os.makedirs(output_dir, exist_ok=True)

img_file = sorted(glob.glob(image_path + "/*"))
seg_file =  sorted(glob.glob(label_path + "/*txt"))

for i,(img,seg) in enumerate(zip(img_file,seg_file)):
    image = cv2.imread(img)
    basename = os.path.basename(img)

    # 画像の縦横サイズ
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
