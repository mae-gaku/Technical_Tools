# import torch
import cv2
from PIL import Image
from pathlib import Path
from numpy import asarray

# 画像をリサイズする関数
def resize_image(img_path, new_size, letterbox=False):
    img = cv2.imread(str(img_path))[:, :, ::-1]
    if letterbox:
        # Letterboxでリサイズする場合
        img_size = img.shape[:2][::-1]
        ratio = min(new_size[0] / img_size[0], new_size[1] / img_size[1])
        new_size = [int(ratio * s) for s in img_size]
        dw = (new_size[0] - img_size[0]) // 2
        dh = (new_size[1] - img_size[1]) // 2
        top, bottom = dh, new_size[1] - img_size[1] - dh
        left, right = dw, new_size[0] - img_size[0] - dw
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        # リサイズする場合
        img = cv2.resize(img, new_size[::-1], interpolation=cv2.INTER_LINEAR)
    return img[:, :, ::-1]

# 画像のリサイズ先サイズを指定
new_size = [1080, 1920]

# 画像が保存されているディレクトリを指定
img_dir = Path('')

# 保存先のディレクトリを指定
output_dir = Path('')
output_dir.mkdir(parents=True, exist_ok=True)

# 画像をリサイズして保存
for img_path in img_dir.glob('*.jpg'):
    img_resized = resize_image(img_path, new_size)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_resized = Image.fromarray(asarray(img_resized))
    img_resized.save(output_dir / img_path.name)
