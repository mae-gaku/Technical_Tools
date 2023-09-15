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


def pose_processing(image,name_num):

    image = letterbox(image, 640, stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    if torch.cuda.is_available():
        image = image.half().to(device)   
    output, _ = model(image)

    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    with torch.no_grad():
        output,xywh = output_to_keypoint(output)  # [batch_id, class_id, x, y, w, h, conf, kpt]
        # print("output",output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    for idx in range(output.shape[0]):
        x_coord, y_coord = plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)    

    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.imshow(nimg)
    plt.savefig("output/" + name_num)


if __name__ == '__main__':
    import glob
    import os

    PATH = "test"
    files = sorted(glob.glob(PATH + "*"))

    for num,file in enumerate(files): 
        name = os.path.splitext(os.path.basename(file))[0] 
        img = cv2.imread(file)
        pose_result = pose_processing(img,name)
    