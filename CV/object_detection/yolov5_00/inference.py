import mmap
import os
import sys
from unittest import result
# sys.path.append(os.path.join(os.path.dirname(__file__), './yolov5'))

from pipeline import Pipeline

def inference(file,img):

    model1 = Pipeline()
    # model1.dataload()
    img = model1.load_img(img)
    pred = model1.infer(img,file)

    result = model1.postprocess(pred)

    
    return result
    

if __name__ == '__main__':
    import glob
    import cv2

    PATH = ''
    files = sorted(glob.glob(PATH + "/*"))

    for i, file in enumerate(files):
        # print(file)
        img = cv2.imread(file)
        out = inference(file,img)
        print(out)

   