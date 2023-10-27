from pathlib import Path
# from requests import JSONDecodeError
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), './yolov5'))
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size,increment_path, non_max_suppression
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import numpy as np
# import json
import cv2
import base64

class Pipeline():

    def __init__(self):
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights= '/tape/best.pt', data= '/tape/data.yaml')
        # model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz=(640, 640), s=self.stride)  # check image size

    def load_img(self, img):

        if type(img).__module__ == np.__name__:
            pass
        else:
            img = base64.b64decode(img)
            img = np.fromstring(img, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)

        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def infer(self,img,file,augment=False,visualize=False):
        img = letterbox(img, self.imgsz, stride=self.stride, auto=True)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        # Run inference
        self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))  # warmup
        # for path, im, im0s, vid_cap, s in self.dataset:
        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim  

        # print(im)

        visualize = increment_path(Path(file).stem, mkdir=True) if visualize else False
        pred = self.model(im, augment=augment, visualize=visualize)

        return pred

    def postprocess(self,pred,classes=None,agnostic_nms=False, max_det=1000, iou_thres=0.45,conf_thres=0.25):
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):  # per image
          
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
              
                label1 = str((self.names[c]))  # add to string
          
                if conf >= 0.70 :
                    print(conf)
                    return ""
                else:
                    print(conf)
                    return ""

           
                

        
    

            