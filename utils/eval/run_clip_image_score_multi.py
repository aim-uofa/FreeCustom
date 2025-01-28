import torch
import clip
import os
from glob import glob
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

method_paths = [
                './multi-concept/ours',
                './multi-concept/cd',
                './multi-concept/perfusion'
                ]

norm = True
method_score = {}
for method_path in method_paths:
    total_score = 0
    
    image_paths = glob(f'{method_path}/*/*g') # All generated images (png/jpg files)

    for image_path in image_paths:
        
        combined_concept_name = image_path.split('/')[3] 
        concept_paths = glob(f'multi-concept/input/{combined_concept_name}/image/*')

        out_image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_features1 = model.encode_image(out_image)

        for concept_path in concept_paths:
            input_image = preprocess(Image.open(concept_path)).unsqueeze(0).to(device)
            image_features0 = model.encode_image(input_image)
            if norm:
                image_features0 /= image_features0.clone().norm(dim=-1, keepdim=True) # normalize feature
                image_features1 /= image_features1.clone().norm(dim=-1, keepdim=True) # normalize feature 
            
            score = (image_features0 @ image_features1.T).mean()
            print(image_path, '\t', concept_path, '\t', score)  
            
        total_score += score.detach()
        
    print(method_path, total_score / 130)
    method_score[f'{method_path}'] = total_score / 130

print(method_score)