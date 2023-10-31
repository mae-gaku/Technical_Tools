import os
import random
import cv2
import albumentations as A
from matplotlib import pyplot as plt
import copy


class Data:
  horizontal_transform = A.Compose([
      A.HorizontalFlip(p=1.0),
  ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

  randomsizedcrop_transform = A.Compose([
      A.RandomSizedCrop(min_max_height=[512, 512], height = 1024,  width=1024, p=1.0),
  ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

  rotate90_transform = A.Compose([
      A.RandomRotate90(p=1.0),
  ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

  rotate180_transform = A.Compose([
      A.RandomRotate90(p=1.0),
      A.RandomRotate90(p=1.0),
  ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

  rotate270_transform = A.Compose([
      A.RandomRotate90(p=1.0),
      A.RandomRotate90(p=1.0),
      A.RandomRotate90(p=1.0),
  ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

  def __init__(self, image="", bboxes=0, id="", class_labels=[]):
    self.image = image
    self.bboxes = bboxes
    self.label = 0
    self.id = id
    self.class_labels = class_labels

  def importdata(self, imgpath):
    dirpath = os.path.dirname(imgpath)[:-7]
    id = os.path.splitext(os.path.basename(imgpath))[0]
    txtpath = dirpath + f"/labels/{id}.txt"
    
    img = cv2.imread(imgpath)
    self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bboxes = []
    with open(txtpath) as f:
      for line in f:
        line_list = line.split(" ")
        bbox = line_list[1:]
        # print("bbox",bbox)
        # print("len(bbox)",len(bbox))
        if len(bbox) >= 5:
            del bbox[4:]
        else:
            pass
        # print("bbox",bbox)
        bbox = [float(i.replace('\n', '')) for i in bbox]
        # print("bbox1",bbox)
        bboxes.append(bbox)
    
    self.bboxes = bboxes
    self.label = 0
    self.id = id
    self.class_labels = ["wheat" for i in range(len(bboxes))]
    
# albumentationsで変換したデータをインポートするメソッド
  def import_transformdata(self, transform_data, origin_data, process):
    self.image = transform_data["image"]
    self.bboxes = transform_data["bboxes"]
    self.label = 0
    self.id = origin_data.id + "_" + process
    self.class_labels = transform_data["class_labels"]
  
# モザイク画像のデータをインポートするメソッド
  def import_mosaicdata(self, img, bboxes, id, class_labels):
    self.image = img
    self.bboxes = bboxes
    self.label = 0
    self.id = id
    self.class_labels = class_labels

    return self.image,self.bboxes, self.label,self.id, self.class_labels

# 左右反転処理したデータを返すメソッド
  def horizonflip(self):
    horizon_transformed = Data.horizontal_transform(image=self.image, 
                                                    bboxes=self.bboxes, 
                                                    class_labels=self.class_labels)
    image = horizon_transformed["image"]
    bboxes = horizon_transformed["bboxes"]
    label = 0
    id = self.id + "_hori"
    class_labels = horizon_transformed["class_labels"]
    horizondata = Data(image, bboxes, id, class_labels)
    return horizondata

# ランダムに切り出してリサイズしたデータを返すメソッド
  def randomsizedcrop(self):
    randomsizedcrop_transformed = Data.randomsizedcrop_transform(image=self.image, 
                                                                 bboxes=self.bboxes, 
                                                                 class_labels=self.class_labels)
    image = randomsizedcrop_transformed["image"]
    bboxes = randomsizedcrop_transformed["bboxes"]
    label = 0
    id = self.id + "_hori"
    class_labels = randomsizedcrop_transformed["class_labels"]
    randomsizedcropdata = Data(image, bboxes, id, class_labels)
    return randomsizedcropdata

# 反時計回りに90ﾟ回転したデータを返すメソッド
  def rotate90(self):
    rotate90_transformed = Data.rotate90_transform(image=self.image, 
                                                   bboxes=self.bboxes, 
                                                   class_labels=self.class_labels)
    image =rotate90_transformed["image"]
    bboxes = rotate90_transformed["bboxes"]
    label = 0
    id = self.id + "_rot90"
    class_labels = rotate90_transformed["class_labels"]
    rot90data = Data(image, bboxes, id, class_labels)
    return rot90data

# 反時計回りに180ﾟ回転したデータを返すメソッド
  def rotate180(self):
    rotate180_transformed = Data.rotate180_transform(image=self.image, 
                                                    bboxes=self.bboxes, 
                                                    class_labels=self.class_labels)
    image =rotate180_transformed["image"]
    bboxes = rotate180_transformed["bboxes"]
    label = 0
    id = self.id + "_rot180"
    class_labels = rotate180_transformed["class_labels"]
    rot180data = Data(image, bboxes, id, class_labels)
    return rot180data

# 反時計回りに270ﾟ回転したデータを返すメソッド
  def rotate270(self):
    rotate270_transformed = Data.rotate270_transform(image=self.image, 
                                                     bboxes=self.bboxes, 
                                                     class_labels=self.class_labels)
    image =rotate270_transformed["image"]
    bboxes = rotate270_transformed["bboxes"]
    label = 0
    id = self.id + "_rot270"
    class_labels = rotate270_transformed["class_labels"]
    rot270data = Data(image, bboxes, id, class_labels)
    return rot270data

# 指定のパスにjpgとtxtファイルでデータ保存するメソッド
  def export_data(self, result_image,result_bboxes, result_label,result_id, result_class_labelsata,imgdirpath):
    id = self.id
    dirpath = imgdirpath[:-7]
    export_imgpath = imgdirpath + f"/{id}.jpg"
    export_txtpath = dirpath + f"/labels/{id}.txt"

    # img = cv2.imread(result_image)
    img = result_image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(export_imgpath, img)
    
    txt = ""
    for bbox in result_bboxes:
      x_min, y_min, width, height = [i for i in bbox]
      line = f"0 {x_min} {y_min} {width} {height}"
      txt += line + "\n"

    f = open(export_txtpath, 'w')
    f.write(txt) 
    f.close()

# 画像とバウンディングボックスを表示するメソッド
  def visualize(self, img_width, img_height, figsize = (10,10)):

    for bbox in self.bboxes:
      x_mid_nor, y_mid_nor, width_nor, height_nor  = [float(i) for i in bbox]

      width = width_nor * img_width  
      height = height_nor * img_height   

      x_min = x_mid_nor * img_width - width/2   
      y_min = y_mid_nor * img_height - height/2    
      x_max = x_min + width
      y_max = y_min + height

      x_min = int(x_min)
      x_max = int(x_max)
      y_min = int(y_min)
      y_max = int(y_max)

      img = cv2.rectangle(self.image,
                          pt1=(x_min, y_min),
                          pt2=(x_max, y_max),
                          color=(255, 0, 0),
                          thickness=3)
      
    plt.figure(figsize = figsize)
    plt.axis('off')
    plt.imshow(img)


# 関数の定義
def generate_mosaicdata(mosaic_group, mode):
  crop_transform = A.Compose([
  A.RandomCrop(height=416, width=416, p=1.0),
  ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))
    
  img_list = [data.image for data in mosaic_group]
  bboxes_list = [data.bboxes for data in mosaic_group]
  classlabels_list = [data.class_labels for data in mosaic_group]
  id_list = [data.id for data in mosaic_group]
  
# ランダムに切り抜く場合の処理
  if not mode == "noncrop":
    pre_transformed_list = [crop_transform(image=img, bboxes=bboxes, class_labels=class_labels) for img, bboxes, class_labels in zip(img_list, bboxes_list, classlabels_list)]
    transformed_list = []

    for crop_data, origin_data in zip(pre_transformed_list, mosaic_group):
      data = Data()
      data.import_transformdata(crop_data, origin_data, "crop")
      transformed_list.append(data)
      
    img1, img2, img3, img4 = [data.image for data in transformed_list]  
    bboxes_list = [data.bboxes for data in transformed_list]
    classlabels_list = [data.class_labels for data in transformed_list]
    mod_id = "_".join([id[:7] for id in id_list])
    
# 切り抜かない場合の処理
  else:
    print("img_list",img_list[0])
    print("img_list1",len(img_list))
    img1, img2, img3, img4 = img_list
    mod_id = "resized_" + "_".join([id[:7] for id in id_list])

  img1_2 = cv2.hconcat([img1, img2])
  img3_4 = cv2.hconcat([img3, img4])
  mod_img = cv2.resize(cv2.vconcat([img1_2, img3_4]), dsize = (1024,1024))

  bboxes1, bboxes2, bboxes3, bboxes4 = bboxes_list
  mod_bboxes1 = []
  for bbox in bboxes1:
    mod_bbox = [i/2 for i in bbox]
    mod_bboxes1.append(mod_bbox)

  mod_bboxes2 = []
  for bbox in bboxes2:
    x, y, width, height = [i/2 for i in bbox]
    mod_x = x + 0.5
    mod_bbox = [mod_x, y, width, height]    
    mod_bboxes2.append(mod_bbox)
  
  mod_bboxes3 = []
  for bbox in bboxes3:
    x, y, width, height = [i/2 for i in bbox]
    mod_y = y + 0.5
    mod_bbox = [x, mod_y, width, height]
    mod_bboxes3.append(mod_bbox)

  mod_bboxes4 = []
  for bbox in bboxes4:
    x, y, width, height = [i/2 for i in bbox]
    mod_x = x + 0.5
    mod_y = y + 0.5
    mod_bbox = [mod_x, mod_y, width, height]
    mod_bboxes4.append(mod_bbox)
  
  mod_bboxes = mod_bboxes1 + mod_bboxes2 + mod_bboxes3 + mod_bboxes4
  mod_classlabels = [cl for classlabels in classlabels_list for cl in classlabels]

  data = Data()
  result_image,result_bboxes, result_label,result_id, result_class_labelsata = data.import_mosaicdata(mod_img, mod_bboxes, mod_id, mod_classlabels)
  
  print(result_image,result_bboxes, result_label,result_id, result_class_labelsata)

  return result_image,result_bboxes, result_label,result_id, result_class_labelsata

def generate_mosaicdatalist(mosaic_ori_dataset, num_mosaicimg, mode = "noncrop"):
  mosaic_dataset = copy.deepcopy(mosaic_ori_dataset)
  mosaic_groups = []

  for i in range(num_mosaicimg):
    mosaic_group = random.sample(mosaic_dataset, 4)
    mosaic_groups.append(mosaic_group)
    mosaic_dataset = list(set(mosaic_group) ^ set(mosaic_dataset)) 
    if len(mosaic_dataset) < 4:
      mosaic_dataset = copy.deepcopy(mosaic_ori_dataset)

  mosaic_data = []
  for mosaic_group in mosaic_groups:
    mosaic_data.append(generate_mosaicdata(mosaic_group, mode))
  
  return mosaic_data
  
if __name__ == '__main__':
    import glob

    img_path = ""
    # img_file = glob.glob(img_path + "/*")
    dataset = []
    i = 0
    
    for filename in os.listdir(img_path):
        fullpath = img_path + "/" + filename
        data = Data()
        data.importdata(fullpath)
        dataset.append(data)
        i +=1
        if i == 4:
            result_image,result_bboxes, result_label,result_id, result_class_labelsata = generate_mosaicdata(dataset, mode = "noncrop")
            # dataset.extend(cropmosaic_dataset)
       

    # for file in img_file:
    #     data.importdata(file)
    #     dataset.append(data)
    #     print("dataset",dataset)
    #     # cropmosaic_dataset = data.generate_mosaicdata(dataset, 400, mode = "crop")
    #     cropmosaic_dataset = generate_mosaicdata(dataset, mode = "crop")
    #     dataset.extend(cropmosaic_dataset)

        # データの可視化（切り抜く場合）
        # datasetを使用して400枚のモザイク画像を作成する。
    # print("cropmosaic_dataset",cropmosaic_dataset[0])
    # dataset[0].visualize(1024, 1024)
            for data in dataset:
                data = Data()
                print(result_image)
                data.export_data(result_image,result_bboxes, result_label,result_id, result_class_labelsata,"/media/sf_virtualbox/My_code/work_nemoto_codes/trash/images2")

