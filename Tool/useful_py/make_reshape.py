import os
import pathlib
import cv2
import numpy as np

dataset_name = "person_personlike_dataset_v2"

work_dir = "/media/sf_virtualbox/person_dataset/"
image_list = os.listdir(work_dir+"/"+dataset_name+"/images")

os.makedirs(f"{work_dir}/mini_dataset/images",exist_ok=True)
os.makedirs(f"{work_dir}/mini_dataset/labels",exist_ok=True)


def scale_box(img, width, height):
    h, w = img.shape[:2]
    aspect = w / h
    if width / height >= aspect:
        nh = height
        nw = round(nh * aspect)
    else:
        nw = width
        nh = round(nw / aspect)

    new_img = cv2.resize(img, dsize=(nw, nh))

    return new_img


def add(img1, img2, out, top, left):

    # height, width = img1.shape[:2]
    # img2[top:height + top, left:width + left] = img1

    cv2.imwrite(out, img2)


process = 0
for i in image_list:
    process += 1
    print(process)
    image = f"{work_dir}/{dataset_name}/images/{i}"
    stem = pathlib.Path(i).stem
    txt = f"{work_dir}/{dataset_name}/labels/{stem}.txt"
    img = cv2.imread(image)
    # height = img.shape[0]
    # width = img.shape[1]
    mini_img = np.array([[[150,150,150]]*854]*480)
    scaled_mini = scale_box(img,854,480)

    mini_width = scaled_mini.shape[1]
    mini_height = scaled_mini.shape[0]

    old_txt = open(txt)

    # process_mini
    if scaled_mini.shape[0]==480:
        add(scaled_mini,mini_img,f"{work_dir}/mini_dataset/images/{i}",0,(854-scaled_mini.shape[1])//2)
        new_txt = open(f"{work_dir}/mini_dataset/labels/{stem}.txt", 'w')
        for line in old_txt:
            tag, x, y, width, height = map(float, line.split())
            new_txt.write(" ".join([str(int(tag)),str((x*mini_width+(854-mini_width)//2)/854),str(y),str((width*mini_width)/854),str(height),'\n']))
        new_txt.close()
    else:
        add(scaled_mini,mini_img,f"{work_dir}/mini_dataset/images/{i}",(480-scaled_mini.shape[0])//2,0)
        new_txt = open(f"{work_dir}/mini_dataset/labels/{stem}.txt", 'w')
        for line in old_txt:
            tag, x, y, width, height = map(float, line.split())
            new_txt.write(" ".join([str(int(tag)),str(x),str((y*mini_height+(480-mini_height)//2)/480),str(width),str((height*mini_height)/480),'\n']))
        new_txt.close()


 