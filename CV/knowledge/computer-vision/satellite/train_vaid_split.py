import glob
import random
import os 
import shutil
import math

images_dir = ""
labels_dir = ''
output_dir = ''

#ランダムで抽出する割合
SAMPLING_RATIO = 0.2
SEED = 0
range_num = 20

def random_sample_file():
    # image_list = []
    # label_list = []
    labels_files = glob.glob(labels_dir + '/*.txt')
    images_files = glob.glob(images_dir + '/*.jpg')
    random.seed(0)
    # x = [random.randint(1,100) for p in range(0,range_num)]

    x = random.sample(range(0,100), k=range_num)
    os.makedirs(output_dir + "/valid_labels",exist_ok=True)
    os.makedirs(output_dir + "/valid_images",exist_ok=True)

    for i in x:
        img = images_files[i]
        labe = labels_files[i]
        # image_list.append(img)
        # label_list.append(labe)

        shutil.move(img,output_dir + "/valid_images/")
        shutil.move(labe ,output_dir + "/valid_labels/")

if __name__ == '__main__':

    random_sample_file()