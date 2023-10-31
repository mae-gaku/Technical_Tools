Run code to perform mosaic augmentation:
```python
python3 main.py --width 832 --height 832 --scale_x 0.5 --scale_y 0.5 --min_area 500 --min_vi 0.5
```

- `--width`: width of mosaic-augmented image
- `--height`: height of mosaic-augmented image
- `--scale_x`: scale_x - scale by width => define width of the top left image
- `--scale_y`: scale_y - scale by height => define height of the top left image
- `--min_area`: min area of box after augmentation we will keep
- `--min_vi`: min area ratio of box after/before augmentation we will keep

