import albumentations as A

import cv2
import numpy as np

image = cv2.imread("path/to/image.jpg")
bboxes = np.array([[10, 20, 100, 200], [50, 70, 150, 250]])  # [x_min, y_min, x_max, y_max]



transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Resize(width=256, height=256, p=1),
], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))

transformed = transform(image=image, bboxes=bboxes, category_ids=[0, 1])
transformed_image = transformed["image"]
transformed_bboxes = np.array(transformed["bboxes"])

for bbox in transformed_bboxes:
    cv2.rectangle(transformed_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

cv2.imshow("Transformed Image", transformed_image)
cv2.waitKey(0)
