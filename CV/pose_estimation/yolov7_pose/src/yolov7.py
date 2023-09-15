import time
import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Pipeline
from utils.pipeline import Pipeline, ModelLoad
from utils.db_analy import Pose


class Inference():
    def __init__(self):
        # load models
        model_loader = ModelLoad()
    
        self.model = model_loader.model
        self.svm_model = model_loader.svm_model

    def inference(self, img, file="./tmp.jpg"):

        # Init pipeline.
        pipeline  = Pipeline()
        pose_db_df = Pose()

        # set models 
        pipeline.model = self.model
        pipeline.svm_model = self.svm_model

        img, img_rgb = pipeline.load_img(img)

        x_coord, y_coord, coord_list_dict, output, xywh, nimg = pipeline.pose_processing(img_rgb)
        
        if len(xywh) == 0 or len(x_coord) == 0 or len(y_coord) == 0:
            pipeline.predicted_result.append([pipeline.not_present_output[0]])
            # return pipeline.gen_response(start_time)
        else:
            for i,j in enumerate(coord_list_dict):
                kpt_coord_dict = pipeline.pose_analy(coord_list_dict[i])

                pose_kpt_db = pose_db_df.pose_db(kpt_coord_dict)

                # svm
                infer_result = pose_db_df.db_svm(pose_kpt_db,pipeline.svm_model)

                infer_result = infer_result.tolist()
                # print("infer_result",infer_result)
                pipeline.predicted_result.append(infer_result)

        output = pipeline.output()

        return  output

if __name__ == '__main__':
    import glob
    import os

    PATH = "File_Path"
    files = sorted(glob.glob(PATH))
    infer = Inference()

    for num,file in enumerate(files): 
        name = os.path.splitext(os.path.basename(file))[0] 
        img = cv2.imread(file)
        pose_result = infer.inference(img,name)
        print("pose_result", pose_result)
    