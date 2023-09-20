import cv2
import numpy as np
import json
import base64
import time

# from PIL import Image

# YOLOv7
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
# import matplotlib.pyplot as plt

# Pytorch-lightning model.
import torch
# import torchvision
import torchvision.transforms as T
import warnings
warnings.filterwarnings('ignore')


class ModelLoad():
    def __init__(self):
        # yolov7
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = "model path"
        weigths = torch.load(self.model_path , map_location=self.device)
        self.model = weigths['model']
        _ = self.model.float().eval()

        if torch.cuda.is_available():
            self.model.half().to(self.device)

        # SVM
        import pickle
        self.svm_model_path = "model path"
        self.svm_model = pickle.load(open(self.svm_model_path, 'rb'))


class Pipeline():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.predicted_result = []

    def load_img(self, img):
        if type(img).__module__ == np.__name__:
            pass
        else:
            img = base64.b64decode(img)
            img = np.fromstring(img, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, img_rgb

    def pose_processing(self,image):
        try:
            image = letterbox(image, 640, stride=64, auto=True)[0]
            image_ = image.copy()
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))

            if torch.cuda.is_available():
                image = image.half().to(self.device) 

            output, _ = self.model(image)

            output = non_max_suppression_kpt(output, 0.50, 0.65, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)
            with torch.no_grad():
                output,xywh = output_to_keypoint(output)  # [batch_id, class_id, x, y, w, h, conf, kpt]
                
            nimg = image[0].permute(1, 2, 0) * 255
            nimg = nimg.cpu().numpy().astype(np.uint8)
            nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

            if not xywh:
                x_coord = []
                y_coord = []
                coord_list = []
                coord_list_dict = []
            else:
                coord_list_dict = []
                for idx in range(output.shape[0]):
                    x_coord, y_coord, coord_list = plot_skeleton_kpts(nimg, output[idx, 7:].T, 3) 
                    coord_list_dict.append(coord_list)
        except Exception as e:
            print(e)

        return x_coord, y_coord, coord_list_dict, output, xywh, nimg

    def pose_analy(self,coord_list):

        key_points = [
                'nose', 'Right eye', 'Left eye', 'Right ear','Left ear', 
                'Right shoulder', 'Left shoulder','Right elbow','Left elbow',
                'Right hand','Left hand','Right hip','Left hip','Right knee','Left knee',
                'Right leg','Left leg']

        kpt_coord_dict = dict(zip(key_points,coord_list))
    
        return kpt_coord_dict

    def output(self):
        out = {

            }

        res_json = json.dumps(out, indent=4)

        return res_json