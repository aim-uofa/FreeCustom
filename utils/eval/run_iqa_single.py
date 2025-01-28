      
from torchmetrics.multimodal import CLIPImageQualityAssessment
import torch
from torchvision.transforms import ToTensor

from PIL import Image

import os

_ = torch.manual_seed(49)
totensor = ToTensor()
metric = CLIPImageQualityAssessment()

path = [
        './single-concept/ours', 
        './single-concept/elite', 
        './single-concept/blip_diffusion',
        './single-concept/dreambooth', 
        './single-concept/neti'
        ]

for root_path in path:
    img_tensor = []
    
    image_paths = os.listdir(root_path)
    for img_path in image_paths:
        img = totensor(Image.open(os.path.join(root_path, img_path))) * 255.
        img_tensor.append(img)
    
    img_tensor = torch.stack(img_tensor, dim=0)
    print(root_path, metric(img_tensor).mean())