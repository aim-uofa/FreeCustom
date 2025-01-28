import os
import torch
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision.transforms import ToTensor

prompt = {
    'dog_flower': 'a dog with flowers around the neck',
    'dog_flowershirts_sunglasses_hats': 'a dog with a blue hat with stripe and sunglasses wearing a white shirt with colorful floral pattern',
    'dog_hat_sunglasses': 'a dog with sunglasses wearing a red and black hat',
    'dog_phone': 'a dog holding a phone',
    'hepburn_sunglasses_beard': 'a girl with beard wearing sunglasses',
    'hinton_cat': 'a man and a cat',
    'hinton_hair': 'a man with curly hair',
    'hinton_spiderman': 'an old man wearing spiderman clothes',
    'man_hairbow_scarf': 'a man wearing a colorful scarf and a colorful hair bow',
    'thanos_beard_flower': 'Thanos with beard holding flowers',
}

_ = torch.manual_seed(42)
metric = CLIPScore(model_name_or_path="./checkpoints/openai/clip-vit-large-patch14")
# metric = CLIPScore(model_name_or_path="./checkpoints/openai/clip-vit-base-patch16")

paths = [
        './multi-concept/ours',
        './multi-concept/cd',
        './multi-concept/perfusion'
        ]

for root_path in paths:
    total_score = 0 
    father_path = os.listdir(root_path)
    for fp in father_path:
        imgs = os.listdir(f'{root_path}/{fp}')
        img_tensor = []
        for img_path in imgs:
            img = ToTensor()(Image.open(f'{root_path}/{fp}/{img_path}')) * 255.
            img_tensor.append(img)
    
        img_tensor=torch.stack(img_tensor, dim=0)
        score = metric(img_tensor, [prompt[fp]]*5)
        print(score)
        total_score += score.detach()
    print(root_path, total_score / 10)