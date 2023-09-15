import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import SamPredictor
from MobileSAM.setup_mobile_sam import setup_model

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

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


if __name__ == "__main__":

    import os
    import glob
    import time
    import memory_profiler as MP
    
    # 処理の開始時刻を記録
    # start_time1 = time.perf_counter()
    
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DEVICE = "cpu"

    # Building MobileSAM predictor
    MOBILE_SAM_CHECKPOINT_PATH = "./mobile_sam.pt"
    checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
    mobile_sam = setup_model()
    mobile_sam.load_state_dict(checkpoint, strict=True)
    mobile_sam.to(device=DEVICE)

    sam_predictor = SamPredictor(mobile_sam)
    
#     end_time1 = time.perf_counter()

#     # 処理時間を計算して表示
#     execution_time1 = end_time1 - start_time1
#     print(f"処理時間: {execution_time1}秒")


    image_path = "/mnt/MobileSAM/dataset/images"
    label_path = "/mnt/MobileSAM/dataset/labels"
#     output_dir = "/mnt/Investigate/result"
    
#     os.makedirs(output_dir, exist_ok=True)
#     os.mkdir(output_dir + "/images")
#     os.mkdir(output_dir + "/labels")

    
    set_ins_id = "on"

    img_file = sorted(glob.glob(image_path + "/*jpg"))
    seg_file =  sorted(glob.glob(label_path + "/*txt"))

    for i,(img,seg) in enumerate(zip(img_file,seg_file)):
        start_time = time.perf_counter()
        b1 = MP.memory_usage()[0]
        
        image = cv2.imread(img)
        basename = os.path.basename(img)
        basename_without_ext = os.path.splitext(os.path.basename(img))[0]


        with open(seg, 'r') as f:
            bbox_str = f.readlines()

        bbox_arrays = []
        cls_id = []
        img_height, img_width,_ = image.shape
        for line in bbox_str:
            bbox_list = line.split()
            class_id = int(bbox_list[0])
            cls_id.append(class_id)
            bbox_float = [float(x) for x in bbox_list[1:]]
            
            x_min_norm = bbox_float[0] - bbox_float[2] / 2
            x_max_norm = bbox_float[0] + bbox_float[2] / 2
            y_min_norm = bbox_float[1] - bbox_float[3] / 2
            y_max_norm = bbox_float[1] + bbox_float[3] / 2

            x_min = int(x_min_norm * img_width)
            y_min = int(y_min_norm * img_height)
            x_max = int(x_max_norm * img_width)
            y_max = int(y_max_norm * img_height)

            bbox_float = [x_min, y_min, x_max, y_max]
            print("bbox_float",bbox_float)
            bbox_array = np.array(bbox_float, dtype=np.float32)
            bbox_arrays.append(bbox_array)

        # convert detections to masks
        masks = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=bbox_arrays
        )
        # print("masks",masks)
        if len(masks) == 0:
            continue
        masks = torch.tensor(masks)
        
        b2 = MP.memory_usage()[0]
        print(b2 - b1)
        
        end_time = time.perf_counter()
        
        # 処理時間を計算して表示
        execution_time = end_time - start_time
        print(f"処理時間: {execution_time}秒")