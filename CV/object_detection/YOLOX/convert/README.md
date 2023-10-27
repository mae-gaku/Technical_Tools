### YAML形式
```
YOLOV5_yaml
├── images
│   ├── train
│   │   ├── images(13).jpg
│   │   └── images(3).jpg
│   └── val
│       ├── images(13).jpg
│       └── images(3).jpg
├── labels
│   ├── train
│   │   ├── images(13).txt
│   │   └── images(3).txt
│   └── val
│       ├── images(13).txt
│       └── images(3).txt
└── sample.yaml
```
```
python3 txt_to_coco.py --yaml_path ./data.yaml
```