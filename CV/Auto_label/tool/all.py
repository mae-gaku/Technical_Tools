import glob
import os
import cv2
import numpy as np
import glob
import os


for idx in range(0,62):
    print("idx",idx)
    image_path = f""
    label_path = f""
    output1_dir = ""
    output2_dir = ""
    output_dir = ""


    images = glob.glob(f"{image_path}/*")
    labels = glob.glob(f"{label_path}/*")
    # print(len(images))
    # print(len(labels))

    for image in images:
        filename = image.split("/")[-1].split(".")[0]

        if not os.path.exists(f"{label_path}/{filename}.txt"):
            os.remove(image)
    # print(len(image))

    for file in labels:
        filename = file.split("/")[-1].split(".")[0]

        if not os.path.exists(f"{image_path}/{filename}.jpg"):
            os.remove(file)
    # print(len(file))


    resize_shape = (1920, 1080)  # リサイズ後の画像のサイズ

    img_file = sorted(glob.glob(image_path + "/*jpg"))
    seg_file = sorted(glob.glob(label_path + "/*txt"))

    for i, (img, seg) in enumerate(zip(img_file, seg_file)):
        image = cv2.imread(img)
        basename = os.path.basename(img)

        # 画像の縦横サイズ
        image_width = image.shape[1]
        
        image_height = image.shape[0]
        # 画像をリサイズ
        resized_image = cv2.resize(image, resize_shape,interpolation=cv2.INTER_AREA)
        image_width = resized_image.shape[1]
        
        image_height = resized_image.shape[0]
        with open(seg, "r") as f:
            segmentation_data_list = f.readlines()

            # リサイズ後のセグメンテーションデータを格納するリスト
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
            output_image_file = os.path.join(output1_dir, os.path.basename(img))
            cv2.imwrite(output_image_file, resized_image)

            # リサイズ後のセグメンテーションデータを保存
            output_seg_file = os.path.join(output2_dir, os.path.splitext(basename)[0] + ".txt")
            with open(output_seg_file, "w") as f:
                f.writelines(resized_segmentation_data_list)

    img_file = sorted(glob.glob(image_path + "/*jpg"))
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
                # print("class_id",class_id)
                points = [float(point) for point in segmentation_data[1:]]
                num_points = len(points) // 2
                points = [(int(points[i * 2] * image_width), int(points[i * 2 + 1] * image_height)) for i in range(num_points)]

                points = np.array(points)
                # print("points",points)
                cv2.polylines(image, [points], True, (0, 255, 0), thickness=2)
                
        cv2.imwrite(output_dir + f"/{i}_" + basename, image)
