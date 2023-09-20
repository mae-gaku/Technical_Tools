import cv2
from utils.pipeline import Pipeline
from utils.pipeline import ModelLoad

class Inference():
    def __init__(self):
        self.load = ModelLoad()

    def inference(self, img):
        pipeline = Pipeline()
        
        # RAM
        pipeline.ram_model = self.load.ram_model
        pipeline.ram_device = self.load.ram_device
        pipeline.transform = self.load.transform
        pipeline.model_ram_img_size = self.load.model_ram_img_size

        ram_img = pipeline.ram_load_img(img)

        out = pipeline.ram_pro(ram_img, img)

        return out


if __name__ == '__main__':
    import glob
    
    Path = "/*"
    files = sorted(glob.glob(Path))

    infer = Inference()

    for file in files:
        img = cv2.imread(file)
        out = infer.inference(img)
        print("out", out)
