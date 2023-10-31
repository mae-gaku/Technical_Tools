# YOLOX

1. git clone & install requirements.txt

```
> git clone https://github.com/Megvii-BaseDetection/YOLOX.git
```

```
> cd YOLOX
```

```
> pip install -U pip && pip install -r requirements.txt

> pip install opencv-python --upgrade

```

2. Download `yolox_s.pt`

```
> wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```


3. Modify the `yolox_x.py` file under `YOLOX/exps/example/custom` as follows

**Please write the name of your dataset**

```
        # Define yourself dataset path
        self.data_dir = "data/custom"
        self.train_ann = "custom_train.json"
        self.val_ann = "custom_val.json"

        self.num_classes = 1

        self.max_epoch = 200
        self.data_num_workers = 4
        self.eval_interval = 50
```


4. Edit the contents of the `coco_classes.py` file under `YOLOX/yolox/data/datasets"`

```
COCO_CLASSES = (  
   "tape_measure",  
 )
```

5. Dataset folder structure

```
YOLOX/data/custom
├── train2017
│   ├──train0.jpg
│   ├──train1.jpg
│   └──train2.jpg  
|
└── val2017
|   ├──train0.jpg
|   ├──train1.jpg
|   └──train2.jpg  
|
└──annotations
    ├──custom_train.json
    └──cusotm_val.json
```

### Training

```
> python3 tools/train.py -f exps/example/custom/yolox_s.py -d 1 -b 16 --fp16 -o -c yolox_s.pth
```

### Inference

```
> python3 tools/demo.py image -f exps/example/custom/yolox_s.py -c "YOLOX_outputs/yolox_s/best_ckpt.pth" --path "data/custom/val2017" --conf 0.3 --nms 0.45 --tsize 640
```