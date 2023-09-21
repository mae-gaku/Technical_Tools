import cv2
import numpy as np
# 画像ファイルパス

PATH_ROOT_DATSET_DIR = "./output"

def convert_yolo2xml(img_path, txt_path, class_file, save_dir_path):
    # try:
    base_image_path = os.path.basename(image_path)
    file_name = os.path.splitext(base_image_path)[0]
    xml_path = f"{PATH_ROOT_DATSET_DIR}/xml"
    # ファイル読み込みa
    image = cv2.imread(img_path)
    height, width, _ = image.shape

    obj_dict = {}
    class_names = []
    class_name_list = []
    with open(class_file,"r") as class_file:
        class_name = class_file.readlines()
        print("class_name",class_name)
        for cls_name in class_name:
            class_names.append(cls_name.strip())
    
    polygon_list = []
    polygons_list = []
    with open(txt_path, "r") as txt_file:
        for line in txt_file:
            content = line.split()
            # content = line.read().split()
            class_name_list.append(class_names[int(content[0])])
            str_list = content[1:]
            float_list = list(map(float, str_list))
            float_set_list = [list(pair) for pair in zip(float_list[0::2], float_list[1::2])]
            xml_float_set_list = yolo2xml_point(float_set_list, height, width)
            polygon_list.append(xml_float_set_list)

    xml = get_xml(file_name, height, width, class_name_list, polygon_list)
    print(f"{save_dir_path}/{file_name}.xml")
    print(f"{PATH_ROOT_DATSET_DIR}/labels/{file_name}.txt")
    with open(f"{save_dir_path}/{file_name}.xml", "w") as file:
        file.write(xml)
    # except Exception as e:
    #     print(e)
    #     return 0

    return 1


def yolo2xml_point(point_set_list, height, width):
    xml_float_set_list = []
    for point_set in point_set_list:
        float_set = []
        x = int(point_set[0] * width)
        y = int(point_set[1] * height)

        float_set.append(x)
        float_set.append(y)
        xml_float_set_list.append(float_set)
    return xml_float_set_list



def get_xml(file_name, height, width, class_name_list, polygon_list):
    import xml.etree.ElementTree as ET
    import xml.dom.minidom

    root = ET.Element("annotation")

    filename = ET.SubElement(root, "filename")
    filename.text = f"{file_name}.jpg"

    folder = ET.SubElement(root, "folder")
    folder.text = ""

    source = ET.SubElement(root, "source")
    sourceImage = ET.SubElement(source, "sourceImage")
    sourceImage.text = ""
    sourceAnnotation = ET.SubElement(source, "sourceAnnotation")
    sourceAnnotation.text = "Datumaro"

    imagesize = ET.SubElement(root, "imagesize")
    nrows = ET.SubElement(imagesize, "nrows")
    nrows.text = str(height)
    ncols = ET.SubElement(imagesize, "ncols")
    ncols.text = str(width)

    idx = 0
    for class_name, polygons in zip(class_name_list, polygon_list):
        object_element = ET.SubElement(root, "object")
        name = ET.SubElement(object_element, "name")
        name.text = class_name

        deleted = ET.SubElement(object_element, "deleted")
        deleted.text = "0"
        verified = ET.SubElement(object_element, "verified")
        verified.text = "0"
        occluded = ET.SubElement(object_element, "occluded")
        occluded.text = "no"
        date = ET.SubElement(object_element, "date")
        date.text = ""

        id = ET.SubElement(object_element, "id")
        id.text = str(idx)
        idx += 1

        parts = ET.SubElement(object_element, "parts")
        hasparts = ET.SubElement(parts, "hasparts")
        hasparts.text = ""
        ispartof = ET.SubElement(parts, "ispartof")
        ispartof.text = ""

        polygon_element = ET.SubElement(object_element, "polygon")
        for set_polygon in polygons:
            pt1_element = ET.SubElement(polygon_element, "pt")
            x1_element = ET.SubElement(pt1_element, "x")
            x1_element.text = str(set_polygon[0])
            y1_element = ET.SubElement(pt1_element, "y")
            y1_element.text = str(set_polygon[1])

    # ET.indent(root, space='  ')
    xml_string = ET.tostring(root, encoding="utf-8")
    dom = xml.dom.minidom.parseString(xml_string)
    pretty_xml = dom.toprettyxml(indent='  ')
    # formatted_xml = pretty_xml.decode("utf-8").replace("><", ">\n<")

    return pretty_xml


import glob
import os

# labels_path = f"{PATH_ROOT_DATSET_DIR}/images"
# labels_path = f"{PATH_ROOT_DATSET_DIR}/labels"
image_dir_path = os.path.join(PATH_ROOT_DATSET_DIR, "images")
txt_dir_path = os.path.join(PATH_ROOT_DATSET_DIR, "labels")
class_file_path = f"{PATH_ROOT_DATSET_DIR}/classes.txt"
save_dir_path = f"{PATH_ROOT_DATSET_DIR}/convert/default"

# mkdir save dir
os.makedirs(save_dir_path, exist_ok=True)
# print(image_dir_path)
# print(txt_dir_path)

# imagesディレクトリ内のファイルパスを取得
image_file_path = []
for root, dirs, files in os.walk(image_dir_path):
    for file in files:
        image_file_path.append(os.path.join(root, file))

# txtディレクトリ内のファイルパスを取得
txt_file_path = []
for root, dirs, files in os.walk(txt_dir_path):
    for file in files:
        txt_file_path.append(os.path.join(root, file))

# 画像ファイルパスと対応するXMLファイルパスをセットで表示
for image_path in image_file_path:
    # 画像ファイル名から拡張子を取り除いてXMLファイル名を生成
    image_filename = os.path.basename(image_path)
    txt_filename = os.path.splitext(image_filename)[0] + ".txt"

    # 対応するXMLファイルのパスを取得
    corresponding_txt_path = os.path.join(txt_dir_path, txt_filename)


    # 画像ファイルパスと対応するXMLファイルパスを表示
    # print("Image file:", image_path)
    # print("XML file:", corresponding_xml_path)
    # print()
    result = convert_yolo2xml(image_path, corresponding_txt_path, class_file_path, save_dir_path)
    if result == 0:
        print("none class")
        continue