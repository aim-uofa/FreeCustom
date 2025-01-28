import os
import torch
import torch.nn as nn
from glob import glob
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# https://medium.com/aimonks/clip-vs-dinov2-in-image-similarity-6fa5aa7ed8c6
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('./checkpoints/huggingface/models/facebook/dinov2-base')
model = AutoModel.from_pretrained('./checkpoints/huggingface/models/facebook/dinov2-base').to(device)

method_paths = [
                './multi-objects/ours',
                './multi-objects/cd',
                './multi-objects/perfusion'
                ]

norm = True
method_score = {}
for method_path in method_paths:
    total_score = 0
    
    image_paths = glob(f'{method_path}/*/*g')

    for image_path in image_paths:
        
        combined_concept_name = image_path.split('/')[3] 
        concept_paths = glob(f'multi-objects/input/{combined_concept_name}/image/*')

        out_image = Image.open(image_path)
        with torch.no_grad():
            inputs2 = processor(images=out_image, return_tensors="pt").to(device)
            outputs2 = model(**inputs2)
            image_features2 = outputs2.last_hidden_state
            image_features2 = image_features2.mean(dim=1)
                
        for concept_path in concept_paths:
            input_image = Image.open(concept_path)
            with torch.no_grad():
                inputs1 = processor(images=input_image, return_tensors="pt").to(device)
                outputs1 = model(**inputs1)
                image_features1 = outputs1.last_hidden_state
                image_features1 = image_features1.mean(dim=1)

            cos = nn.CosineSimilarity(dim=0)
            sim = cos(image_features1[0],image_features2[0]).item()
            sim = (sim + 1) / 2
            print(image_path, '\t', concept_path, '\t', sim) 
            
            score = sim 
            total_score += score
        
    print(method_path, total_score / 130)
    method_score[f'{method_path}'] = total_score / 130

print(method_score)