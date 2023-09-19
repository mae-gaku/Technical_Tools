import sys
from os.path import dirname, abspath
SRC_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(SRC_DIR)

import cv2
import numpy as np
import time
# Pipeline
from utils.pipeline import Pipeline
from utils.pipeline import ModelLoad


class Inference():
    def __init__(self):
        # Loading model only once when deploying
        self.loader = ModelLoad()


    def inference(self, img, out, threshold, file="./tmp.jpg"):
        pipeline = Pipeline()

        pipeline.model1 = self.loader.model1
        pipeline.model1_class_names = self.loader.model1_class_names
        pipeline.model1_threshold = threshold
        if threshold is None:
            pipeline.model1_threshold = self.loader.model1_threshold
        pipeline.model1_iou_threshold = self.loader.model1_iou_threshold

        # Loading image.
        img, _ = pipeline.load_img(img)

        # Object detection 1st.
        cls_bboxes1, cls_id_list1 = pipeline.detect_object(img,out)

        # Getting class images list and class boxes list
        cls_img_list1, cls_box_list1 = pipeline.get_cls_image_list(img, cls_bboxes1, cls_id_list1)

        if len(cls_img_list1) == 0:
            return


if __name__ == '__main__':
    # (Debug on local)
    import time
    start = time.time()
    from utils.definitions import get_project_root_path
    PROJECT_ROOT = get_project_root_path()
    import glob

    infer = Inference()

    cap = cv2.VideoCapture('./person.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))//10000
    print("fps",fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter('./lmodel.mp4', fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res_json = infer.inference(frame,out , None)
    end = time.time()  
    print("itme:",end - start)  
    out.release()
    cap.release()
    cv2.destroyAllWindows()
