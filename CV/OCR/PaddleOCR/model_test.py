from cv2 import line
from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(rec_model_dir='./en_PP-OCRv3_rec/', rec_char_dict_path='./custom_en_dict_v2.txt') # need to run only once to download and load model into memory
img_path = './001.jpg'
result = ocr.ocr(img_path, cls=False)

for line in result:
    print(line)
    
# draw result
from PIL import Image
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
