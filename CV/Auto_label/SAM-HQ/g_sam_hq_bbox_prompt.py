import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm

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


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    # ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)
    
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

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    # parser.add_argument(
    #     "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    # )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )

    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--input_label", type=str, required=True, help="path to label file")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    parser.add_argument("--set_ins_id", type=str, required=True, help="path to label file")

    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    # sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path_1 = args.input_image
    output_dir = args.output_dir
    device = args.device
    label_path = args.input_label
    set_ins_id = args.set_ins_id

    import glob
    
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    os.mkdir(output_dir + "/images")
    os.mkdir(output_dir + "/labels")
    os.mkdir(output_dir + "/grad")

    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    # else:
    #     predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    image_path_1 = sorted(glob.glob(image_path_1 + "/*jpg"))
    label_path_1 = sorted(glob.glob(label_path + "/*txt"))

    for x,(image_path,labels_path) in enumerate(zip(image_path_1,label_path_1)):
        a = cv2.imread(image_path)
        width = a.shape[1]
        height = a.shape[0]
        image_pil, image = load_image(image_path)

        basename = os.path.basename(image_path)
        print(basename)
        basename_without_ext = os.path.splitext(os.path.basename(image_path))[0]
        
        with open(labels_path, 'r') as f:
            bbox_str = f.readlines()

        bbox_arrays = []
        cls_id = []
        for line in bbox_str:
            bbox_list = line.split()
            class_id = int(bbox_list[0])
            cls_id.append(class_id)
            bbox_float = [float(x) for x in bbox_list[1:]]
            bbox_array = np.array(bbox_float, dtype=np.float32)
            bbox_arrays.append(bbox_array)

        boxes_filt = torch.from_numpy(np.array(bbox_arrays))

        if len(boxes_filt) == 0:
            continue
        
        image_pil.save(os.path.join(output_dir + "/images/", basename))
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cuda()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
                
        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )
        torch.cuda.empty_cache()
        
        
        if len(masks) == 0:
            continue
        with open(os.path.join(output_dir + "/labels/", basename_without_ext + ".txt"), 'w') as f:
                ins_id = 0
                for idxm, (mask1,id) in enumerate(zip(masks,cls_id)):
                    merge_masks = []
                    result = []
                    # Convert the mask to a binary image
                    binary_mask = mask1.squeeze().cpu().numpy().astype(np.uint8)

                    # Find the contours of the mask
                    contours, hierarchy = cv2.findContours(binary_mask,
                                                        cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_SIMPLE)
                    if set_ins_id == "on":
                        if len(contours) > 1:
                            ins_id +=1

                        # Get the segmentation mask for object 
                        for largest_contour in contours:
                            # Get the segmentation mask for object 
                            segmentation = largest_contour.flatten().tolist()

                            mask=segmentation

                            # convert mask to numpy array of shape (N,2)
                            points = np.array(mask).reshape((-1, 2)).astype(np.int32)

                            normalized_points = points / np.array([width, height])[np.newaxis, :]

                            class_id = id  

                            if len(contours) > 1:
                                instance_id = ins_id
                                yolo_data = f"{class_id} {instance_id} {' '.join(normalized_points.flatten().astype(str))}\n"
                                f.write(yolo_data)

                            else:
                                yolo_data = f"{class_id} {' '.join(normalized_points.flatten().astype(str))}\n"
                                f.write(yolo_data)
                                
                    else:
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

                        class_id = id  

                        yolo_data = f"{class_id} {' '.join(normalized_points.flatten().astype(str))}\n"
                        f.write(yolo_data)

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        predictor.reset_image()
        print("Complete")




