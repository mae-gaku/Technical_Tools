import cv2
from argparse import ArgumentParser
import glob
import numpy as np

def main(source, width, height):
    
    img_dir = source
    images = glob.glob(img_dir + "/*")

    for image in images:
        base_img = np.zeros((height, width, 3), np.uint8)
        img = cv2.imread(image)
        img_h, img_w = img.shape[:2]
        aspect_height = height / img_h
        aspect_width = width / img_w

        if aspect_width < aspect_height:
            new_size = (int(img_w * aspect_width), int(img_h * aspect_width))
        else:
            new_size = (int(img_w * aspect_height), int(img_h * aspect_height))
        
        new_img = cv2.resize(img, dsize=new_size)
        base_img[int(height / 2 - new_size[1] / 2):int(height / 2 + new_size[1] / 2), int(width / 2 - new_size[0] / 2):int(width / 2 + new_size[0] / 2), :] = new_img
        cv2.imwrite(image, base_img)


def parse_opt():
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default="./images", help="image folder to resize")
    parser.add_argument("--width", type=int, default=300, help="put image width in pixels")
    parser.add_argument("--height", type=int, default=300, help="put image height in pixels")

    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))
