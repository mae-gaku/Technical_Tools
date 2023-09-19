# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path
import sys
import os

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams, scale_coords,clip_boxes
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from openvino.runtime import Core, Layout



def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xyxy, conf, class_num,count):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()
    
    # cv2.rectangle(img, (x1,y1), (x2, y2), (255, 191, 0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    # for i in range(5):
    #     point_x = int(landmarks[2 * i])
    #     point_y = int(landmarks[2 * i + 1])
    #     cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    # label = str(conf)[:5]
    label = f"face{count}"
    age = "30-40"
    gender = "female"
    age_label = f"Age:{age}"
    gender_label = f"Gender:{gender}"


    # cv2.putText(img, label, (x1 + 60, y1 + 20), 0, tl / 2, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    # cv2.putText(img, conf, (x1 + 60, y1 + 40), 0, tl / 2, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


    mix_rate=0.1
    color=(255, 191, 0)
    thickness=2
    length_rate=0.3

    # add rectangles
    ann_img = img.copy()
    xyxy = [xyxy]
    for bbox in xyxy:
        cv2.rectangle(ann_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, -1)

    img = cv2.addWeighted(ann_img, mix_rate, img, 1-mix_rate, 0)


    # add corner lines
    ann_img = img.copy()
    for bbox in xyxy:
        xmin, ymin, xmax, ymax = bbox
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        line_length = int(min(xmax-xmin, ymax-ymin) * length_rate)

        # top-left
        cv2.line(ann_img, (xmin, ymin), (xmin+line_length, ymin), color, thickness)
        cv2.line(ann_img, (xmin, ymin), (xmin, ymin+line_length), color, thickness)
        # top-right
        cv2.line(ann_img, (xmax, ymin), (xmax-line_length, ymin), color, thickness)
        cv2.line(ann_img, (xmax, ymin), (xmax, ymin+line_length), color, thickness)
        # bottom-left
        cv2.line(ann_img, (xmin, ymax), (xmin+line_length, ymax), color, thickness)
        cv2.line(ann_img, (xmin, ymax), (xmin, ymax-line_length), color, thickness)
        # bottom-right
        cv2.line(ann_img, (xmax, ymax), (xmax-line_length, ymax), color, thickness)
        cv2.line(ann_img, (xmax, ymax), (xmax, ymax-line_length), color, thickness)

    mix_rate += 0.3
    img = cv2.addWeighted(ann_img, mix_rate, img, 1-mix_rate, 0)
    cv2.putText(img, label, (x1 + 60, y1 + 20), 0, tl / 2, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
    cv2.putText(img, age_label, (x1 + 60, y1 + 40), 0, tl / 2, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
    cv2.putText(img, gender_label, (x1 + 60, y1 + 60), 0, tl / 2, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

    # cv2.imwrite('bird_out.jpg', img)

    return img



def detect(
    model,
    source,
    device,
    project,
    name,
    exist_ok,
    save_img,
    view_img
):
    # Load model
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.45
    imgsz=(640, 640)
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    is_file = Path(source).suffix[1:] in (img_formats + vid_formats)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    
    # Dataloader
    if webcam:
        print('loading streams:', source)
        dataset = LoadStreams(source, img_size=imgsz)
        bs = 1  # batch_size
    else:
        print('loading images', source)
        dataset = LoadImages(source, img_size=imgsz)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    for path, im, im0s, vid_cap in dataset:
        
        if len(im.shape) == 4:
            orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis= 0)
        else:
            orgimg = im.transpose(1, 2, 0)
        
        orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        # imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
        imgsz = (640,640)
        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1).copy()
        img = np.array(img, dtype=np.float)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = np.expand_dims(img, 0) 
 

        # img = img.transpose(2, 0, 1).copy()

        # img = torch.from_numpy(img).to(device)
        # img = img.float()  # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)

        pred = list(model([img]).values())
        
        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            
            p = Path(p)  # to Path
            save_path = str(Path(save_dir) / p.name)  # im.jpg

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class

                # det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()
                det = torch.from_numpy(det)
                count = 1
                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    print("conf",conf)
                    # landmarks = det[j, 5:15].view(-1).tolist()
                    # class_num = det[j, 15].cpu().numpy()
                    class_num = det[j, 5].cpu().numpy()
                    print("class_num",class_num)
                    
                    im0 = show_results(im0, xyxy, conf, class_num,count)
                    count +=1
                
            if view_img:
                cv2.imshow('result', im0)
                k = cv2.waitKey(1)
                    
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    try:
                        vid_writer[i].write(im0)
                    except Exception as e:
                        print(e)

                    
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-img', action='store_true', help='save results')
    parser.add_argument('--view-img', action='store_true', help='show results')
    opt = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = load_model(opt.weights, device)
    ie = Core()

    model_path = "./yolov5s-face"
    network = ie.read_model(model=model_path + '.xml', weights=model_path + '.bin')
    network.get_parameters()[0].set_layout(Layout("NCHW"))

    model = ie.compile_model(network, device_name="CPU")
 
    detect(model, opt.source, device, opt.project, opt.name, opt.exist_ok, opt.save_img, opt.view_img)
