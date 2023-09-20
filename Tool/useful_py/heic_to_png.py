from PIL import Image
import pyheif

def conv(image_path):
    new_name = image_path.replace('HEIC', 'png')
    heif_file = pyheif.read(image_path)
    data = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
        )
    data.save(new_name, "PNG")

import glob
lst = glob.glob("HEIC path")
for l in lst:
    conv(l)