import cv2
import numpy as np
import time
# Pipeline
from utils.pipeline import Pipeline
from utils.pipeline import ModelLoad


class Inference():
    def __init__(self):
        self.load = ModelLoad()

    def inference(self, img, out, threshold, file="./tmp.jpg"):
        pipeline = Pipeline()

        pipeline.model1 = self.load.model1
        pipeline.model1_class_names = self.load.model1_class_names
        pipeline.model1_threshold = threshold
        if threshold is None:
            pipeline.model1_threshold = self.load.model1_threshold
        pipeline.model1_iou_threshold = self.load.model1_iou_threshold

        bboxes, cls_id_list = pipeline.detect_face(img,out)

        return bboxes, cls_id_list


if __name__ == '__main__':

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
    out.release()
    cap.release()
    cv2.destroyAllWindows()
