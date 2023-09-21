import cv2
import numpy as np
import base64
import torch
import warnings
warnings.filterwarnings('ignore')

## RAM
from Tag2Text.models import tag2text
from Tag2Text import inference_ram
import torchvision.transforms as TS
import torch
import torchvision.transforms as transforms


class ModelLoad():
    def __init__(self):
        ## RAM
        model_ram_path = "./weight/ram_swin_large_14m.pth"
        self.model_ram_img_size = 384
        self.ram_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load model
        self.ram_model = tag2text.ram(pretrained=model_ram_path,
                                            image_size=384,
                                            vit='swin_l')
        self.ram_model.eval()
        self.ram_model = self.ram_model.to(self.ram_device)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((self.model_ram_img_size, self.model_ram_img_size)),
            transforms.ToTensor(), self.normalize
        ])


class Pipeline():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("self.device", self.device)

    def ram_load_img(self, img):
        from PIL import Image   
        pil_image = Image.fromarray(img[..., ::-1])
        raw_image = pil_image.convert("RGB").resize(
        (self.model_ram_img_size,self.model_ram_img_size))
        raw_image = self.transform(raw_image).unsqueeze(0).to(self.ram_device)

        return raw_image
    
    def ram_pro(self,img, c_img):

        res, _ , rate = inference_ram.inference(img , self.ram_model)
        tags = [tag.strip() for tag in res[0].split('|') if tag.strip()]
        search_strings = [] 

        matching_indices = []
        matching_labels = []

        for search_string in search_strings:
            indices = [index for index, label in enumerate(tags) if search_string in label]
            matching_indices.extend(indices)
            matching_labels.extend([tags[index] for index in indices])

        unique_matching_indices = list(set(matching_indices))
        unique_matching_labels = list(set(matching_labels))

        if len(unique_matching_indices) == 0 or len(unique_matching_labels) == 0:
            self.ram_result_list.append("")
        
        else:
            rate_list = [tensor[1].item() for tensor in rate]
            selecte_rate = [rate_list[idx] for idx in unique_matching_indices]
            rate_dict = {label: element for label, element in zip(unique_matching_labels, selecte_rate)}

            label = search_strings.copy()
            labels_rate_list = []
            for lab in label:
                if lab in rate_dict:
                    corresponding_element = rate_dict[lab]
                    labels_rate_list.append(corresponding_element)
                else:
                    labels_rate_list.append(0)

        
        return 