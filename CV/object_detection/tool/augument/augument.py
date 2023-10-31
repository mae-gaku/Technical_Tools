# augument.py  
import glob, cv2, pathlib, time, os
from utils.ano_utils import generate_xml, save_with_rectangle, pascalvoc2bbox, generate_xml
import albumentations as A


# custom transformer
def augmentation(h, w, p=0.5, pad_size=640):
    tranform = A.Compose([
        # # ----- flip ----------
        # A.HorizontalFlip(p=p),
        # A.VerticalFlip(p=p),

        # # ----- skew ----------
        # A.Perspective(p=p),
        # A.Affine(p=p, fit_output=True, mode=0),     # caution: image size will increase with tilt.
        
        # # ----- rotate, padding ----------
        # A.RandomRotate90(p=p),
        # A.Rotate(p=p, limit=30, border_mode=0),
        # # A.PadIfNeeded(p=p, min_height=pad_size, min_width=pad_size, border_mode=0),

        # # ----- emboss ----------
        # A.Emboss(p=p, alpha=(0.2, 1.0), strength=(0.2, 1.0)),
        
        # # ----- blur, sharpen ----------
        # A.Blur(p=p),
        # A.MedianBlur(p=p),
        # A.Sharpen(p=p),

        # # ----- noise ----------
        # A.GaussNoise(p=p, var_limit=(10, 200)),
        
        # # ----- color & contrust & brightness ----------
        # A.ToGray(p=p),
        # A.CLAHE(p=p),
        # A.RandomBrightnessContrast(p=p),
        # A.RandomGamma(p=p),
        # A.HueSaturationValue(p=p),

        # ----- crop ----------
        # A.RandomCrop(p=p, width=int(w*0.8), height=int(h*0.8)),
        # A.CenterCrop(p=p, width=int(w*0.8), height=int(h*0.8)),
        A.RandomSizedBBoxSafeCrop(p=p, width=int(w*0.8), height=int(h*0.8)),

        # ----- weather ----------
        # A.RandomSnow(p=p),
        # A.RandomRain(p=p),
        # A.RandomSunFlare(p=p, src_radius=int(min(h, w)*0.2)),
        # A.RandomShadow(p=p, shadow_roi=(0, 0, 1, 1), num_shadows_upper=10, shadow_dimension=10),

    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1, label_fields=['class_labels']))

    return tranform



if __name__ == '__main__':
    save_dir = ''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir + '/images/', exist_ok=True)
        os.makedirs(save_dir + '/labels/', exist_ok=True)
    
    img_file_path = ""
    label_file_path = ""
    img_file = glob.glob(img_file_path)
    labels_file = glob.glob(label_file_path)
    for img_path,label_path in zip(img_file,labels_file):
        # for i in range(10):
        img = cv2.imread(img_path)
        cls_name_list,  bboxes = pascalvoc2bbox(pathlib.Path(label_path).with_suffix('.xml'))

        # save original image
        save_with_rectangle(img.copy(), bboxes, 0)

        # convert 
        conv = augmentation(*img.shape[:2])(image=img, bboxes=bboxes, class_labels=cls_name_list)
        conv_img = conv['image']
        conv_bboxes = conv['bboxes']
        conv_class_labels = conv['class_labels']
        
        # xml
        doc = generate_xml(conv_img, pathlib.Path(path).name, conv_bboxes, conv_class_labels)
        
        # save augmented image and annotation
        random_file_name = str(time.time()).replace('.', '')
        cv2.imwrite(f'{save_dir}/images/{random_file_name}.jpg', conv_img)
        with open(f'{save_dir}/labels/{random_file_name}.xml', 'w') as f:
            doc.writexml(f, encoding='utf-8', newl='\n', indent='', addindent='\t')        
