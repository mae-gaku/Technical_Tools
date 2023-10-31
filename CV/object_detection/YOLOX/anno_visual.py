import cv2
import glob
import os


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


def bbox_(img, bboxes, x):

    mix_rate=0.1
    color=(255, 191, 0)
    # color=(0, 191, 255)
    thickness=2
    length_rate=0.3
    ann_img = img.copy()
    for bbox in bboxes:
        cv2.rectangle(ann_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, -1)
    img = cv2.addWeighted(ann_img, mix_rate, img, 1-mix_rate, 0)
    # add corner lines
    ann_img = img.copy()
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        
        line_length = int(min(xmax-xmin, ymax-ymin) * length_rate)
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        # top-left
        cv2.line(ann_img, (xmin, ymin), (xmin+line_length, ymin), color, thickness)
        cv2.line(ann_img, (xmin, ymin), (xmin, ymin+line_length), color, thickness)
        # top-right
        cv2.line(ann_img, (xmax, ymin), (xmax-line_length, ymin), color, thickness)
        cv2.line(ann_img, (xmax, ymin), (xmax, ymin+line_length), color, thickness)
        # bottom-left
        cv2.line(ann_img, (xmin, ymax), (xmin+line_length, ymax), color, thickness)
        cv2.line(ann_img, (xmin, ymax), (xmin, ymax-line_length), color, thickness)
        # bottom-right
        cv2.line(ann_img, (xmax, ymax), (xmax-line_length, ymax), color, thickness)
        cv2.line(ann_img, (xmax, ymax), (xmax, ymax-line_length), color, thickness)

    mix_rate += 0.3
    img = cv2.addWeighted(ann_img, mix_rate, img, 1-mix_rate, 0)
    cv2.imwrite(f"./{x}.jpg", img)



if __name__ == "__main__":

    image_path = "./images"
    label_path = "./labels"
    output_image_dir = "./result_image"
    output_label_dir = "./result_label"
    
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_path_1 = sorted(glob.glob(image_path + "/*jpg"))
    label_path_1 = sorted(glob.glob(label_path + "/*txt"))
    
    for x,(image_path,labels_path) in enumerate(zip(image_path_1,label_path_1)): 
        basename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        with open(labels_path, 'r') as f:
            bbox_str = f.readlines()

        bbox_lists = []
        cls_id = []
        for line in bbox_str:
            bbox_list = line.split()
            class_id = int(bbox_list[0])
            cls_id.append(class_id)
            bbox_float = [float(x) for x in bbox_list[1:]]
            bbox = yolo_to_bbox(bbox_float, w, h)
            bbox_lists.append(bbox)
            print("bbox_list", bbox_lists)

        # show
        # for i, box in enumerate(bbox_lists):
        #     xmin, ymin, xmax, ymax = box
        #     print("box", box)
        #     cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
        #     cv2.putText(
        #         image,
        #         text="person",
        #         org=(int(xmin), int(ymin)),
        #         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #         fontScale=0.7,
        #         color=(0, 0, 255),
        #     )
        # cv2.imwrite(f"./result_image/{x}.jpg", image)


        
        bbox_(image, bbox_lists, x)
