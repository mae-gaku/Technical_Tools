import os
import xml.etree.ElementTree as ET

def convert_pascal_to_yolo(xml_path, output_dir):
    # クラス名の対応を定義
    class_mapping = {
        "car_(automobile)": 0
    }
    
    # XMLファイルを解析
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 画像の幅と高さを取得
    size = root.find("size")
    width = float(size.find("width").text)
    height = float(size.find("height").text)
    
    # 出力用のテキストファイルパスを生成
    txt_filename = os.path.splitext(os.path.basename(xml_path))[0] + ".txt"
    txt_path = os.path.join(output_dir, txt_filename)
    
    with open(txt_path, "w") as txt_file:
        # 各オブジェクトについて処理
        for obj in root.findall("object"):
            # オブジェクトのクラス名を取得
            class_name = obj.find("name").text
            
            # "body"クラスのみを処理
            if class_name == "Shoes":
                # オブジェクトのバウンディングボックスを取得
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                
                # バウンディングボックスの中心座標と幅・高さを計算
                x_center = (xmin + xmax) / (2 * width)
                y_center = (ymin + ymax) / (2 * height)
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height
                
                # YOLO形式のテキスト行を生成
                # class_id = class_mapping[class_name]
                class_id = 3
                yolo_line = f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"
                
                # テキストファイルに書き込み
                txt_file.write(yolo_line)

            elif class_name == "Helmet":
                # オブジェクトのバウンディングボックスを取得
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                
                # バウンディングボックスの中心座標と幅・高さを計算
                x_center = (xmin + xmax) / (2 * width)
                y_center = (ymin + ymax) / (2 * height)
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height
                
                # YOLO形式のテキスト行を生成
                # class_id = class_mapping[class_name]
                class_id = 0
                yolo_line = f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"
                
                # テキストファイルに書き込み
                txt_file.write(yolo_line)

            elif class_name == "Vest":
                # オブジェクトのバウンディングボックスを取得
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                
                # バウンディングボックスの中心座標と幅・高さを計算
                x_center = (xmin + xmax) / (2 * width)
                y_center = (ymin + ymax) / (2 * height)
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height
                
                # YOLO形式のテキスト行を生成
                # class_id = class_mapping[class_name]
                class_id = 1
                yolo_line = f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"
                
                # テキストファイルに書き込み
                txt_file.write(yolo_line)
            
            elif class_name == "Gloves":
                # オブジェクトのバウンディングボックスを取得
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                
                # バウンディングボックスの中心座標と幅・高さを計算
                x_center = (xmin + xmax) / (2 * width)
                y_center = (ymin + ymax) / (2 * height)
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height
                
                # YOLO形式のテキスト行を生成
                # class_id = class_mapping[class_name]
                class_id = 2
                yolo_line = f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"
                
                # テキストファイルに書き込み
                txt_file.write(yolo_line)
            
            elif class_name == "harness":
                # オブジェクトのバウンディングボックスを取得
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                
                # バウンディングボックスの中心座標と幅・高さを計算
                x_center = (xmin + xmax) / (2 * width)
                y_center = (ymin + ymax) / (2 * height)
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height
                
                # YOLO形式のテキスト行を生成
                # class_id = class_mapping[class_name]
                class_id = 4
                yolo_line = f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"
                
                # テキストファイルに書き込み
                txt_file.write(yolo_line)


xml_dir = ""
output_dir = ""

for filename in os.listdir(xml_dir):
    if filename.endswith(".xml"):
        xml_path = os.path.join(xml_dir, filename)
        convert_pascal_to_yolo(xml_path, output_dir)
