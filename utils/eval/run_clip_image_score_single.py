import torch
import clip
import os
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

method_paths = [
                './single-concept/ours', 
                './single-concept/elite', 
                './single-concept/blip_diffusion',
                './single-concept/dreambooth', 
                './single-concept/neti'
                ]
norm = True
method_score = {}
for method_path in method_paths:
    total_score = 0
    
    image_paths = os.listdir(f'{method_path}/')
    for image_path in image_paths:
        
        input_image = preprocess(Image.open(f'./single-concept/input/{image_path.split(" ")[0]}.png')).unsqueeze(0).to(device)
        out_image   = preprocess(Image.open(f'{method_path}/{image_path}')).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features0 = model.encode_image(input_image)
            image_features1 = model.encode_image(out_image)
            
            if norm:
                image_features0 /= image_features0.clone().norm(dim=-1, keepdim=True) # normalize feature
                image_features1 /= image_features1.clone().norm(dim=-1, keepdim=True) # normalize feature
            
            score = (image_features0 @ image_features1.T).mean()
            print(image_path, '\t', score) 
            
        total_score += score.detach()
        
    print(method_path, total_score / 200)
    method_score[f'{method_path}'] = total_score / 200

print(method_score)