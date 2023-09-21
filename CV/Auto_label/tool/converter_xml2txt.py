import numpy as np

import cv2
import numpy as np
# 画像ファイルパス

def convert_xml2yolo(img_path, xml_path, class_path):
    base_image_path = os.path.basename(image_path)
    file_name = os.path.splitext(base_image_path)[0]
    txt_path = "./txt"
    classes = []
    # read image path
    image = cv2.imread(img_path)
    # read classes.txt
    with open(class_path, "r") as file:
        for line in file:
            classes.append(line.strip())

    # parse xml
    xml_dicts = parse_xml(xml_path)
    if xml_dicts == 0:
        return 0
    # print("xml dict: ", xml_dict)

    for obj_id, obj_info in xml_dicts.items():

        # 画像の縦横サイズ
        image_width = obj_info["imagesize"][1]
        image_height = obj_info["imagesize"][0]

        seg_data = obj_info["polygon"]

        # ポイントをx, yペアに分割する
        points = np.array(seg_data).reshape((-1, 2)).astype(np.int32)
        normalized_points = points / np.array([image_width, image_height])[np.newaxis, :]

        # yolo形式のセグメンテーションデータに変換する
        class_id = classes.index(obj_info["name"])  # クラスIDは0とする
        yolo_data = f"{class_id} {' '.join(normalized_points.flatten().astype(str))}\n"

        # テキストファイルに書き込む
        print(f"{txt_path}/{file_name}.txt")
        with open(f"{txt_path}/{file_name}.txt", "a") as f:
            f.write(yolo_data)


    return 1



def parse_xml(xml_path):
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    try:
        xml_dicts = {}
        for obj in root.findall('object'):
            obj_id = obj.find('id').text
            name = obj.find('name').text
            polygon = []
            if name == "person":
                pass
            else:
                for pt in obj.findall('.//pt'):
                    x = float(pt.find('x').text)
                    y = float(pt.find('y').text)
                    polygon.extend([x, y])

                imagesize = root.find('.//imagesize')
                nrows = int(imagesize.find('nrows').text)
                ncols = int(imagesize.find('ncols').text)

                xml_dicts[obj_id] = {
                    'imagesize': [nrows, ncols],
                    'name': name,
                    'polygon': polygon
                }
    except Exception as e:
        print("skip")
        print(e)
        print(xml_path)
        return 0


    return xml_dicts


import glob
import os
PATH_ROOT_DATSET_DIR = "dataset name"
image_dir_path = os.path.join(PATH_ROOT_DATSET_DIR, "images")
xml_dir_path = os.path.join(PATH_ROOT_DATSET_DIR, "labels")
class_path = f"{PATH_ROOT_DATSET_DIR}/classes.txt"

# imagesディレクトリ内のファイルパスを取得
image_file_path = []
for root, dirs, files in os.walk(image_dir_path):
    for file in files:
        image_file_path.append(os.path.join(root, file))

# xmlディレクトリ内のファイルパスを取得
xml_file_path = []
for root, dirs, files in os.walk(xml_dir_path):
    for file in files:
        xml_file_path.append(os.path.join(root, file))

# 画像ファイルパスと対応するXMLファイルパスをセットで表示
for image_path in image_file_path:
    # 画像ファイル名から拡張子を取り除いてXMLファイル名を生成
    image_filename = os.path.basename(image_path)
    xml_filename = os.path.splitext(image_filename)[0] + ".xml"

    # 対応するXMLファイルのパスを取得
    corresponding_xml_path = os.path.join(xml_dir_path, xml_filename)

    result = convert_xml2yolo(image_path, corresponding_xml_path, class_path)
    if result == 0:
        print("none class")
        continue
#    break