import argparse
import os
import numpy as np
import torch
from PIL import Image

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)
import cv2
import numpy as np
import glob
import json
from utils.definitions import get_project_root_path, get_config_path
PROJECT_ROOT = "/mnt/hq-sam/Grounded-Segment-Anything/"
CONFIG_PATH = "/mnt/hq-sam/Grounded-Segment-Anything/config.json"

class ModelLoad():
    def __init__(self):
        # Loading setting json file.
        with open(CONFIG_PATH) as f:
            config = json.load(f)

        # grounding dino
        grounding_model_config = config["grounding_dino_model"]
        config_file =  grounding_model_config["config_path"]
        grounded_checkpoint = PROJECT_ROOT + grounding_model_config["model_path"]
        self.device = 'cuda'
        self.text_threshold = grounding_model_config["text_threshold"]
        self.bbox_threshold = grounding_model_config["bbox_threshold"]
        self.text_prompt = grounding_model_config["text_prompt"]
        args = SLConfig.fromfile(config_file)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(grounded_checkpoint, map_location=self.device)
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        _ = model.eval()
        # ground_model = load_model(config_file, grounded_checkpoint, device=self.device)
        self.ground_model = model.to('cuda:0')

        # load model(SAM)
        sam_model_config = config["sam_model"]
        sam_checkpoint = PROJECT_ROOT + sam_model_config["sam_model_path"]
        sam_hq_checkpoint = PROJECT_ROOT + sam_model_config["sam_hq_model_path"]
        use_sam_hq = sam_model_config["use_sam_hq"]

        if use_sam_hq:
            self.predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(self.device))
        else:
            self.predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(self.device))

        # dataset
        dataset_config = config["dataset"]
        self.input_image = dataset_config["input_image"]
        self.output_dir = dataset_config["output_dir"]
        self.cls_id = dataset_config["cls_id"]

        # make dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.mkdir(self.output_dir + "/images")
        os.mkdir(self.output_dir + "/labels")
        os.mkdir(self.output_dir + "/grad")

# class Inference():
#     def __init__(self):


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    width, height = image_pil.size
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image, width, height


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

    
def merge_multi_segment(segments):
    """Merge multi segments to one list.
    Find the coordinates with min distance between each segment,
    then connect these coordinates with one thin line to merge all 
    segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...], 
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]
    
    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def min_index(arr1, arr2):
    """Find a pair of indexes with the shortest distance. 
    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


if __name__ == "__main__":

    piplie = ModelLoad()
    original_image_path = sorted(glob.glob(piplie.input_image + "/*"))

    for _,image_path in enumerate(original_image_path):
        image_pil, image, width, height  = load_image(image_path)

        basename = os.path.basename(image_path)
        basename_without_ext = os.path.splitext(os.path.basename(image_path))[0]

        image_pil.save(os.path.join(piplie.output_dir + "/images/", basename))
        
        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            piplie.ground_model, image, piplie.text_prompt, piplie.bbox_threshold, piplie.text_threshold, device=piplie.device
        )
        
        if len(boxes_filt) == 0:
            continue
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        piplie.predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = piplie.predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(piplie.device)

        masks, _, _ = piplie.predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(piplie.device),
            multimask_output = False,
        )
     
        if len(masks) == 0:
            continue

        with open(os.path.join(piplie.output_dir + "/labels/", basename_without_ext + ".txt"), 'w') as f:
            ins_id = 0
            for idxm, mask1 in enumerate(masks):
                merge_masks = []
                result = []

                # Convert the mask to a binary image
                binary_mask = mask1.squeeze().cpu().numpy().astype(np.uint8)

                # Find the contours of the mask
                contours, hierarchy = cv2.findContours(binary_mask,
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
                for largest_contour  in contours:
                    segmentation = largest_contour.flatten().tolist()
                    mask=segmentation
                    merge_masks.append(mask)

                if len(contours) > 1:
                    segment_label = merge_multi_segment(merge_masks)
                    for seg_mask in segment_label:
                        for x, y in seg_mask:
                            result.append(int(x))
                            result.append(int(y))
                    mask = result
                else:
                    mask = merge_masks[0]

                # convert mask to numpy array of shape (N,2)
                points = np.array(mask).reshape((-1, 2)).astype(np.int32)

                normalized_points = points / np.array([width, height])[np.newaxis, :]

                class_id = piplie.cls_id  

                yolo_data = f"{class_id} {' '.join(normalized_points.flatten().astype(str))}\n"
                f.write(yolo_data)

        piplie.predictor.reset_image()

