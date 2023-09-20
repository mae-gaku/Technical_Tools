import torch
import warnings
warnings.filterwarnings('ignore')

# blip
from blip.models.blip import load_checkpoint
from blip.models.blip_itm import BLIP_ITM
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

class ModelLoad():
    def __init__(self):
        # BLIP
        model_blip_path = "./model/blip/weights/"
        device = torch.device('cpu')
        self.model_blip_class_names = []
        model = BLIP_ITM(image_size=384, vit='base')
        self.model_blip,_ = load_checkpoint(model,model_blip_path)
        self.model_blip.eval()
        self.model_blip = self.model_blip.to(device=device)

class Pipeline():
    def __init__(self):
        self.device = torch.device('cpu')
        self.blip_result_list = []
    
    def blip_load_image(self, image, image_size=384,device='cpu'):
        from PIL import Image
        raw_image = Image.fromarray(image)
        transform = transforms.Compose([
            transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        image = transform(raw_image).unsqueeze(0).to(device)   

        return image


    def classify_blip(self,img):

        score = []
        for caption in self.model_blip_class_names:
            itm_output = self.model_blip(img,caption,match_head='itm')
            itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
            score.append(itm_score)

        score_max = max(score)
        max_index = score.index(score_max)

        print("Inference Result:", self.model_blip_class_names[max_index])
        
        self.blip_result_list.append(self.model_blip_class_names[max_index])

