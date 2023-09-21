import cv2
import numpy as np
import glob
import os

image_path = "./images"
label_path = "./labels"
output1_dir = "./result_image"
output2_dir = "./result_label"
resize_shape = (960, 639)  # リサイズ後の画像のサイズ

os.makedirs(output1_dir, exist_ok=True)
os.makedirs(output2_dir, exist_ok=True)

img_file = sorted(glob.glob(image_path + "/*jpg"))
seg_file = sorted(glob.glob(label_path + "/*txt"))

for i, (img, seg) in enumerate(zip(img_file, seg_file)):
    image = cv2.imread(img)
    basename = os.path.basename(img)

    # 画像の縦横サイズ
    image_width = image.shape[1]
    
    image_height = image.shape[0]

    resized_image = cv2.resize(image, resize_shape,interpolation=cv2.INTER_AREA)
    image_width = resized_image.shape[1]
    
    image_height = resized_image.shape[0]
    with open(seg, "r") as f:
        segmentation_data_list = f.readlines()

        resized_segmentation_data_list = []

        for segmentation_data in segmentation_data_list:
            segmentation_data = segmentation_data.split()
            if len(segmentation_data) == 0:
                break

            class_id = int(segmentation_data[0])

            # 座標をリサイズ
            points = [float(point) for point in segmentation_data[1:]]
            num_points = len(points) // 2
            points = [(points[i * 2], points[i * 2 + 1]) for i in range(num_points)]

            # リサイズ後の座標を計算
            resized_points = []
            for point in points:
                resized_x = float(point[0] * resize_shape[0] / image_width)
                resized_y = float(point[1] * resize_shape[1] / image_height)
                if resized_x < 0:
                    resized_x = 0
                elif resized_x >= resize_shape[0]:
                    resized_x = resize_shape[0] - 1
                if resized_y < 0:
                    resized_y = 0
                elif resized_y >= resize_shape[1]:
                    resized_y = resize_shape[1] - 1

                resized_points.append((resized_x, resized_y))

            # リサイズ後のセグメンテーションデータを作成
            new_segmentation_data = f"{class_id}"
            for point in resized_points:
                
                new_segmentation_data += f" {point[0]} {point[1]}"
            new_segmentation_data += "\n"
            resized_segmentation_data_list.append(new_segmentation_data)

            # リサイズ後の画像を保存
        output_image_file = os.path.join(output1_dir, f"{i}_" + os.path.basename(img))
        cv2.imwrite(output_image_file, resized_image)

        # リサイズ後のセグメンテーションデータを保存
        output_seg_file = os.path.join(output2_dir, f"{i}_" + os.path.splitext(basename)[0] + ".txt")
        with open(output_seg_file, "w") as f:
            f.writelines(resized_segmentation_data_list)
