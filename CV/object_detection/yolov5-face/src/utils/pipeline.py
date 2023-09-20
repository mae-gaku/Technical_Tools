import cv2
import numpy as np
import json
import base64
import time
from utils.general import check_img_size,non_max_suppression_face, scale_coords

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
        network = ie.read_model(model='./model/yolov5/weights/yolov5l-face.xml', weights='./model/yolov5/weights/yolov5l-face.bin')
        network.get_parameters()[0].set_layout(Layout("NCHW"))

        self.model1 = ie.compile_model(network, device_name="CPU")


class Pipeline():
    def __init__(self):
        self.device = torch.device('cpu')

    def detect_face(self, input_img,out):
        bboxes, class_id_list = [], []

        imgsz = (640,640)
        img = letterbox(input_img, new_shape=imgsz)[0]
        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1).copy()
        img = np.array(img, dtype=np.float)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = np.expand_dims(img, 0) 

        pred = list(self.model1([img]).values())
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

            label = f'face'
            color = (0, 0, 255)
            xyxy = bboxes[j]
            xyxy = [int(i) for i in xyxy]
            cv2.rectangle(input_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness=2)
            cv2.putText(input_img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(input_img)

        return bboxes, class_id_list
    