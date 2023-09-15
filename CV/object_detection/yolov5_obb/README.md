# YOLOv5-obb

yolov5-obb

- https://github.com/hukaixuan19970627/yolov5_obb

set up

```
git clone https://github.com/hukaixuan19970627/yolov5_obb.git
cd /yolov5_obb/
pip install -r requirements.txt

```

```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

```
cd utils/nms_rotated
python setup.py develop
```


### Train

```
python3 train.py --weights yolov5s.pt --data data.yaml --epochs 10 --img 640 --device cpu --batch-size 8

```

### Detect 

```
python3 detect.py --weights best.pt --source test_images --imgsz 640 --device cpu 

```

# anotation tool

### roLabelImg

- https://github.com/cgvict/roLabelImg

set up
```
sudo apt-get install pyqt5-dev-tools
make qt5py3
python3 roLabelImg.py
```

shot cut key
z : 回転（逆時計周り）
c : 回転（時計回り）
