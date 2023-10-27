import cv2
import numpy as np
import glob
import os
import argparse

def yolo_to_bbox(yolo_format, image_width, image_height):
    cx, cy, w, h = yolo_format
    x = int((cx - (w / 2)) * image_width)
    y = int((cy - (h / 2)) * image_height)

    x_max = int((cx + (w / 2)) * image_width)
    y_max = int((cy + (h / 2)) * image_height)

    class_index = 0

    return [x, y, x_max, y_max]

def show_box(boxes, wight, height):
    yolo_boxes = []
    for box in boxes:
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        
        x_center /= wight
        y_center /= height
        box_width /= wight
        box_height /= height
        
        yolo_box = f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        yolo_boxes.append(yolo_box)
    
    return yolo_boxes


def resize_image_label(img_files, seg_files, output_image_dir, output_label_dir, resize_size):
    for i, (img, seg) in enumerate(zip(img_files, seg_files)):
        image = cv2.imread(img)
        basename = os.path.basename(img)

        image_width = image.shape[1]
        image_height = image.shape[0]

        if image_width > image_height:
            new_width = resize_size
            new_height = int(image_height * (resize_size / image_width))
        else:
            new_width = int(image_width * (resize_size / image_height))
            new_height = resize_size

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        rh, rw, _ = resized_image.shape

        with open(seg, 'r') as f:
            bbox_str = f.readlines()

        bbox_lists = []
        cls_id = []
        for line in bbox_str:
            bbox_list = line.split()
            class_id = int(bbox_list[0])
            cls_id.append(class_id)
            bbox_float = [float(x) for x in bbox_list[1:]]
            bbox = yolo_to_bbox(bbox_float, rw, rh)
            bbox_lists.append(bbox)
            print("bbox_list", bbox_lists)



            output_seg_file = os.path.join(output_label_dir, f"{i}_" + os.path.splitext(basename)[0] + ".txt")
            yolo_boxes = show_box(bbox_lists, rw, rh)
            with open(output_seg_file, "w") as f:
                for yolo_box in yolo_boxes:
                    f.write(yolo_box + "\n")

            # output_image_file = os.path.join(output_image_dir, f"{i}_" + os.path.basename(img))
            # cv2.imwrite(output_image_file, resized_image)

        for i, box in enumerate(bbox_lists):
            xmin, ymin, xmax, ymax = box
            print("box", box)
            cv2.rectangle(resized_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
            cv2.putText(
                resized_image,
                text="person",
                org=(int(xmin), int(ymin)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
            )

        cv2.imwrite(f"/media/sf_virtualbox/Academy_person/result_image/{i}.jpg", resized_image)


def drew_ploygon(img_file, seg_file, output_dir):

    for i,(img,seg) in enumerate(zip(img_file,seg_file)):
        image = cv2.imread(img)
        basename = os.path.basename(img)

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


if __name__ == "__main__":

    image_path = "/media/sf_virtualbox/Academy_person/images"
    label_path = "/media/sf_virtualbox/Academy_person/labels"
    output_image_dir = "/media/sf_virtualbox/Academy_person/result_image"
    output_label_dir = "/media/sf_virtualbox/Academy_person/result_label"
    
    # resize_size = size
    resize_size = 960
    # on or off
    set_drew = "on"

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    img_file = sorted(glob.glob(image_path + "/*jpg"))
    seg_file = sorted(glob.glob(label_path + "/*txt"))


    resize_image_label(img_file, seg_file, output_image_dir, output_label_dir, resize_size)

    # if set_drew == "on":
    #     image_path = output_image_dir
    #     label_path = output_label_dir
    #     output_dir = "/media/sf_virtualbox/groundingdino_sam/drew_result"

    #     os.makedirs(output_dir, exist_ok=True)

    #     img_file = sorted(glob.glob(image_path + "/*jpg"))
    #     seg_file =  sorted(glob.glob(label_path + "/*txt"))

    #     drew_ploygon(img_file, seg_file, output_dir)

