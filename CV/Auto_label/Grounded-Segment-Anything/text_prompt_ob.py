import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases
    

def show_box(boxes, wight, height):
    yolo_boxes = []
    for box in boxes:
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        
        x_center /= wight
        y_center /= height
        box_width /= wight
        box_height /= height
        
        yolo_box = f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        yolo_boxes.append(yolo_box)
    
    return yolo_boxes

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--cls_id", type=int, required=True, help="path to label file")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    image_path_1 = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    cls_id = args.cls_id

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # print(output_dir + "/images")
    os.mkdir(output_dir + "/images")
    os.mkdir(output_dir + "/labels")
    os.mkdir(output_dir + "/grad")

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    model = model.to('cuda:0')
    
    import glob
    image_path_1 = sorted(glob.glob(image_path_1 + "/*jpg"))
    for x,image_path in enumerate(image_path_1):
        a = cv2.imread(image_path)
        width = a.shape[1]
        height = a.shape[0]
        image_pil, image = load_image(image_path)

        basename = os.path.basename(image_path)
        print("basename",basename)
        basename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
        image_pil.save(os.path.join(output_dir + "/images/", basename))
        
        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )

        if len(boxes_filt) == 0:
            continue
            
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        
        yolo_boxes = show_box(boxes_filt, width, height)
        # print("yolo_boxes",yolo_boxes)
        # with open(os.path.join(output_dir + "/labels/", basename_without_ext + ".txt"), "w") as f:
        #     for yolo_box in yolo_boxes:
        #         f.write(yolo_box + "\n")
        color = (0, 255, 0)  # BGR形式の色 (ここでは緑)
        thickness = 2 
        for line,label in zip(yolo_boxes,pred_phrases):
            bbox_list = line.split()
            bbox_float = [float(x) for x in bbox_list[1:]]
            
            x_min_norm = bbox_float[0] - bbox_float[2] / 2
            x_max_norm = bbox_float[0] + bbox_float[2] / 2
            y_min_norm = bbox_float[1] - bbox_float[3] / 2
            y_max_norm = bbox_float[1] + bbox_float[3] / 2

            x_min = int(x_min_norm * W)
            y_min = int(y_min_norm * H)
            x_max = int(x_max_norm * W)
            y_max = int(y_max_norm * H)
            
            # print("pred_phrases", pred_phrases)
            cv2.rectangle(a, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)

        
        cv2.imwrite(output_dir + "/grad/"+ f"{x}.jpg" , a)

            