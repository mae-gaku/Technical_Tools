from ultralytics import YOLO

from utils.tools import *

import argparse

import ast





def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--model_path", type=str, default="./weights/FastSAM.pt", help="model"

    )

    parser.add_argument(

        "--img_path", type=str, default="./images/dogs.jpg", help="path to image file"

    )

    parser.add_argument("--imgsz", type=int, default=1024, help="image size")

    parser.add_argument(

        "--iou",

        type=float,

        default=0.9,

        help="iou threshold for filtering the annotations",

    )

    parser.add_argument(

        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'

    )

    parser.add_argument(

        "--conf", type=float, default=0.4, help="object confidence threshold"

    )

    parser.add_argument(

        "--output", type=str, default="./output/", help="image save path"

    )

    parser.add_argument(

        "--randomcolor", type=bool, default=True, help="mask random color"

    )

    parser.add_argument(

        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"

    )

    parser.add_argument(

        "--point_label",

        type=str,

        default="[0]",

        help="[1,0] 0:background, 1:foreground",

    )

    parser.add_argument("--box_prompt", type=str, default="[0,0,0,0]", help="[x,y,w,h]")

    parser.add_argument(

        "--better_quality",

        type=str,

        default=False,

        help="better quality using morphologyEx",

    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument(

        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"

    )

    parser.add_argument(

        "--retina",

        type=bool,

        default=True,

        help="draw high-resolution segmentation masks",

    )

    parser.add_argument(

        "--withContours", type=bool, default=False, help="draw the edges of the masks"

    )

    parser.add_argument("--input_image", type=str, required=True, help="path to image file")

    parser.add_argument("--input_label", type=str, required=True, help="path to image file")

    parser.add_argument(

        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"

    )

    return parser.parse_args()





def main(args,model, image_path,labels_path):

    

    a = cv2.imread(image_path)

    width = a.shape[1]

    height = a.shape[0]



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

        x = int((bbox_float[0] - bbox_float[2] / 2) * width)

        y = int((bbox_float[1] - bbox_float[3] / 2) * height)

        w = int(bbox_float[2] * width)

        h = int(bbox_float[3] * height)



        bbox_arrays.append(x)

        bbox_arrays.append(y)

        bbox_arrays.append(w)

        bbox_arrays.append(h)





    args.box_prompt = bbox_arrays



    results = model(

        image_path,

        imgsz=args.imgsz,

        device=args.device,

        retina_masks=args.retina,

        iou=args.iou,

        conf=args.conf,

        max_det=100,

    )



    if args.box_prompt[2] != 0 and args.box_prompt[3] != 0:

        annotations = prompt(results, image_path, args, box=True)

        annotations = np.array([annotations])



        if isinstance(annotations, torch.Tensor):

            annotations = annotations.cpu().numpy()



        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_h = image.shape[0]

        original_w = image.shape[1]



        with open(os.path.join(args.output_dir + "/labels/", basename_without_ext + ".txt"), 'w') as f:

            for i, mask in enumerate(annotations):

                merge_masks = []

                result = []

                

                if type(mask) == dict:

                    mask = mask["segmentation"]

                annotation = mask.astype(np.uint8)

                if args.retina == False:

                    annotation = cv2.resize(

                        annotation,

                        (original_w, original_h),

                        interpolation=cv2.INTER_NEAREST,

                    )

                contours, hierarchy = cv2.findContours(

                    annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE

                )



                for largest_contour  in contours:

                    segmentation = largest_contour.flatten().tolist()

                    mask=segmentation



                    merge_masks.append(mask)

                if len(contours) > 1:

                    segment_label = merge_multi_segment(merge_masks)

                    # print("segment_label",segment_label)

                    for seg_mask in segment_label:

                        # print("seg_mask",seg_mask)

                        for x, y in seg_mask:

                            result.append(int(x))

                            result.append(int(y))



                    mask = result

                else:

                    mask = merge_masks[0]



                # convert mask to numpy array of shape (N,2)

                points = np.array(mask).reshape((-1, 2)).astype(np.int32)



                normalized_points = points / np.array([width, height])[np.newaxis, :]



                class_id = 0  



                yolo_data = f"{class_id} {' '.join(normalized_points.flatten().astype(str))}\n"

                f.write(yolo_data)







def prompt(results, image_path, args, box=None, point=None, text=None):

    ori_img = cv2.imread(image_path)

    ori_h = ori_img.shape[0]

    ori_w = ori_img.shape[1]

    if box:

        mask, idx = box_prompt(

            results[0].masks.data,

            convert_box_xywh_to_xyxy(args.box_prompt),

            ori_h,

            ori_w,

        )

    else:

        return None

    return mask





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

    import glob

    args = parse_args()

    image_path_1 = sorted(glob.glob(args.input_image + "/*jpg"))

    label_path_1 = sorted(glob.glob(args.input_label + "/*txt"))

    import time
    start_time = time.perf_counter()
    model = YOLO(args.model_path)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print("処理時間：", execution_time, "秒")

    

    os.makedirs(args.output_dir, exist_ok=True)

    os.mkdir(args.output_dir + "/images")

    os.mkdir(args.output_dir + "/labels")



    for x,(image_path,labels_path) in enumerate(zip(image_path_1,label_path_1)):
        
        main(args, model, image_path,labels_path)
        
        
