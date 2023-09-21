import os
import xml.etree.ElementTree as ET

def is_not_empty(xml_path):
    # # XMLファイルを解析
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        print("XMLファイルの解析中にエラーが発生しました。")
        return False
    root = tree.getroot()
    
    with open(xml_path, "w") as txt_file:
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name == 'fire_extinguisher':
                return True
    return False
        # XMLファイルを解析して要素の数を確認
    # try:
    #     tree = ET.parse(xml_path)
    #     root = tree.getroot()
    #     num_elements = len(root)
    #     if num_elements == 0:
    #         return True
    #     else:
    #         return False
    # except ET.ParseError:
    #     print("XMLファイルの解析中にエラーが発生しました。")
    #     return False

if __name__ == "__main__":
    import glob
    image_path = "./labels"

    # img_file = sorted(glob.glob(image_path + "/*xml"))
    # print("img_file",img_file)
    
    for file_path in os.listdir(image_path):
        xml_path = os.path.join(image_path, file_path)
        if is_not_empty(xml_path):
            print("XMLファイルに要素が含まれています。")
             
        else:
            print("XMLファイルは空です。")
            os.remove(xml_path)
            
