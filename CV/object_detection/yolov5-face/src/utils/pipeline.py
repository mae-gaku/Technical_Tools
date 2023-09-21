import cv2
import numpy as np
import json
import base64
import time
<<<<<<< HEAD

# Yolov5
# from utils.augmentations import letterbox
from utils.general import check_img_size,non_max_suppression_face, scale_coords

# Pytorch-lightning model.
=======
from utils.general import check_img_size,non_max_suppression_face, scale_coords

>>>>>>> master
import torch
import warnings
warnings.filterwarnings('ignore')
from utils.datasets import letterbox

from openvino.runtime import Core, Layout


class ModelLoad():
    def __init__(self):
        model1_threshold = 0.3
        self.model1_iou_threshold = 0.45

        ie = Core()
        # model_path = "/home/gaku/yolov5-face/yolov5s-face"
<<<<<<< HEAD
        network = ie.read_model(model='./modules/yolov5/weights/yolov5l-face.xml', weights='./modules/yolov5/weights/yolov5l-face.bin')
=======
        network = ie.read_model(model='./model/yolov5/weights/yolov5l-face.xml', weights='./model/yolov5/weights/yolov5l-face.bin')
>>>>>>> master
        network.get_parameters()[0].set_layout(Layout("NCHW"))

        self.model1 = ie.compile_model(network, device_name="CPU")


<<<<<<< HEAD
class VIPipeline():
    def __init__(self):
        self.device = torch.device('cpu')

        # Flask output.
        self.height = 0
        self.width = 0
        self.display_bbox_list = []
        self.pred_list = []
        self.display_cls_name_list = []
        self.display_comment_list = []


    def load_img(self, img):
        if type(img).__module__ == np.__name__:
            # (Debug on local)
            pass
        else:
            img = base64.b64decode(img)
            img = np.fromstring(img, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, img_rgb


    def detect_object(self, input_img,out):
        # Execute yolov5 detection.
        # Input: RGB(input_img)
        start_time = time.perf_counter()
        bboxes, class_id_list = [], []

        # try:
=======
class Pipeline():
    def __init__(self):
        self.device = torch.device('cpu')

    def detect_face(self, input_img,out):
        bboxes, class_id_list = [], []

>>>>>>> master
        imgsz = (640,640)
        img = letterbox(input_img, new_shape=imgsz)[0]
        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1).copy()
        img = np.array(img, dtype=np.float)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = np.expand_dims(img, 0) 
<<<<<<< HEAD
 
        # Inference
        pred = list(self.model1([img]).values())

        # Apply NMS
=======

        pred = list(self.model1([img]).values())
>>>>>>> master
        pred = non_max_suppression_face(pred, self.model1_threshold, self.model1_iou_threshold)

        for i, det in enumerate(pred):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], input_img.shape).round()

        det = torch.from_numpy(det)
        i = 0
        for j in range(det.size()[0]):
            bboxe = det[j, :4].view(-1).tolist()
            bboxes.append(bboxe)
            self.pred_list += det[j, 4].cpu().numpy()
            class_id_list.append(0)
<<<<<<< HEAD
            self.display_cls_name_list.append(self.model1_config["class_names"][0])
=======
>>>>>>> master

            label = f'face'
            color = (0, 0, 255)
            xyxy = bboxes[j]
            xyxy = [int(i) for i in xyxy]
            cv2.rectangle(input_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness=2)
            cv2.putText(input_img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(input_img)

<<<<<<< HEAD
        self.time_object_detection += time.perf_counter() - start_time

        return bboxes, class_id_list


    def get_cls_image_list(self, img, class_bboxes, class_id_list):
        class_image_list = []
        class_box_list = []
        for box, _ in zip(class_bboxes, class_id_list):
            class_img, class_box = self.crop_img_by_box(img, box)
            class_image_list.append(class_img)
            class_box_list.append(class_box)

        return class_image_list, class_box_list


    def crop_img_by_box(self, img, bbox, margin=False):
        # Image cropping with bounding box.
        xmin, ymin, xmax, ymax = bbox

        if margin:
            h, w = ymax-ymin, xmax-xmin
            xmin = max(xmin, 0)
            ymin = max(ymin+int(0.09*h), 0)
            xmax = min(xmax+int(0*w), img.shape[1])
            ymax = min(ymax-int(*h), img.shape[0])

        return img[int(ymin):int(ymax), int(xmin):int(xmax)], [int(xmin), int(ymin), int(xmax), int(ymax)]


    def to_json(self):
        self.height = 0
        self.width = 0
        self.display_bbox_list = []
        self.pred_list = []
        self.display_cls_name_list = []
        self.display_comment_list = []

        # return res_json
=======
        return bboxes, class_id_list
    
>>>>>>> master
