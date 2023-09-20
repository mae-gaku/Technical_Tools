import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
model = weigths['model']
_ = model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)

def pose_processing(cap):

    while(True):
        ret, frame = cap.read()
        ## logic here
        image = letterbox(frame, 960, stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        if torch.cuda.is_available():
            image = image.half().to(device)   
        output, _ = model(image)

        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        with torch.no_grad():
            output = output_to_keypoint(output)  # [batch_id, class_id, x, y, w, h, conf, kpt]
            # print("output",output)
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

        for idx in range(output.shape[0]):
            x_coord, y_coord = plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)    


        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import glob
    import os
    cap = cv2.VideoCapture("")
    pose_processing(cap)

    # PATH = "test/"
    # files = sorted(glob.glob(PATH + "*"))

    # for num,file in enumerate(files): 
    #     name = os.path.splitext(os.path.basename(file))[0] 
    #     img = cv2.imread(file)

    


