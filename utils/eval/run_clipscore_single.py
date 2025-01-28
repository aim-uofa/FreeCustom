      
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchmetrics.multimodal.clip_score import CLIPScore

_ = torch.manual_seed(42)
metric = CLIPScore(model_name_or_path="./checkpoints/openai/clip-vit-large-patch14")
# metric = CLIPScore(model_name_or_path="./checkpoints/openai/clip-vit-base-patch16")

method_paths = [
                './single-concept/ours', 
                './single-concept/elite', 
                './single-concept/blip_diffusion',
                './single-concept/dreambooth', 
                './single-concept/neti'
                ]

method_score = {}
for method_path in method_paths:
    total_score = 0
    
    image_paths = os.listdir(f'{method_path}/')
    for image_path in image_paths:
        image = ToTensor()(Image.open(f'{method_path}/{image_path}')) * 255. #! * 255. is very important

        text = image_path.split('"')[1]
        text = text.replace('sys', '')
        text = text.replace('a photo of ', ' ')
        text = text.replace('{}', image_path.split(' ')[0].replace('0', '').replace('1', '').replace('2', ''))
        text = text.replace('S', image_path.split(' ')[0].replace('0', '').replace('1', '').replace('2', ''))
        
        score = metric(image, text)
        print(image_path, '\t', text, '\t', score)
        total_score += score.detach()
        
    print(method_path, total_score / 200)
    method_score[f'{method_path}'] = total_score / 200

print(method_score)