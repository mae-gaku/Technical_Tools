import cv2

img = cv2.imread()
mosaic_labels = []
height, width = img.shape[0:2]

for i, row in enumerate(mosaic_labels):
    x,y,w,h = row[1:]
    x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
    x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)