import cv2

def yolo_to_bbox(yolo_format, image_width, image_height):
    cx, cy, w, h = yolo_format
    x = int((cx - (w / 2)) * image_width)
    y = int((cy - (h / 2)) * image_height)

    x_max = int((cx + (w / 2)) * image_width)
    y_max = int((cy + (h / 2)) * image_height)

    class_index = 0

    return [x, y, x_max, y_max]

def resize_image(image, new_width, new_height):
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

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


if __name__ == "__main__":
    import glob
    import os

    image_path = "/media/sf_virtualbox/Academy_person/images"
    label_path = "/media/sf_virtualbox/Academy_person/labels"
    output_image_dir = "/media/sf_virtualbox/Academy_person/result_image"
    output_label_dir = "/media/sf_virtualbox/Academy_person/result_label"
    
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_path_1 = sorted(glob.glob(image_path + "/*jpg"))
    label_path_1 = sorted(glob.glob(label_path + "/*txt"))
    
    new_width = 300
    new_height = 200

    for x,(image_path,labels_path) in enumerate(zip(image_path_1,label_path_1)): 
        basename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        resized_image = resize_image(image, new_width, new_height)

        rh, rw, _ = resized_image.shape


        with open(labels_path, 'r') as f:
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


        yolo_boxes = show_box(bbox_lists, rw, rh)
        with open(os.path.join(output_label_dir + "/", basename_without_ext + ".txt"), "w") as f:
            for yolo_box in yolo_boxes:
                f.write(yolo_box + "\n")

    # show
        # for i, box in enumerate(bbox_lists):
        #     xmin, ymin, xmax, ymax = box
        #     print("box", box)
        #     cv2.rectangle(resized_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
        #     cv2.putText(
        #         resized_image,
        #         text="person",
        #         org=(int(xmin), int(ymin)),
        #         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #         fontScale=1,
        #         color=(0, 0, 255),
        #     )

        # cv2.imwrite(f"/media/sf_virtualbox/Academy_person/result_image/{x}.jpg", resized_image)
