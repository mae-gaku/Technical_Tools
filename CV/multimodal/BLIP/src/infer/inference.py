import cv2
from utils.pipeline import Pipeline
from utils.pipeline import ModelLoad


class Inference():
    def __init__(self):
        self.load = ModelLoad()

    def inference(self, img):
        pipeline = Pipeline()

        pipeline.model_blip = self.load.model_blip
        pipeline.model_blip_class_names = self.load.model_blip_class_names


        imgs = pipeline.blip_load_image(img)
        out = pipeline.classify_blip(imgs)

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