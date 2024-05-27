from sentence_transformers import SentenceTransformer, util
from PIL import Image
import argparse
import sys
import os
import re

def remove_numbers(token):
 
  return re.sub(r"\d", "", token)

def get_last_token(text):
  tokens = text.rsplit("_", 1)  # Split from the right at most once
  if len(tokens) == 1:
    return text  # No underscore found, return the entire string
  else:
    return tokens[-1]

def extract_text(path):
    # skip bad results
    if 'bad' in path or 'dbg' in path:
        return 'skip'
    obj = remove_numbers(get_last_token(path))
    if obj[-1] =='s':
        #plural        
        text = "A DSLR photo of " + obj
    else:
        text = "A DSLR photo of a " + obj
    
    return text

def extract_full_path(opt, folder):
    # if not 200 take 100
    base_path = os.path.join(opt.results, folder)
    latest = "ep0200"
    full_path = f'{base_path}/validation/df_{latest}_0005_{opt.mode}.png'
    if os.path.exists(full_path):
        return full_path
    else:
        latest = "ep0100"
        full_path = f'{base_path}/validation/df_{latest}_0005_{opt.mode}.png'
        if os.path.exists(full_path):
            return full_path
        else:
            print(f"Path {full_path} does not exist")
            exit(1)

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default="A DSLR photo of a dog", type=str, help="text prompt")
    parser.add_argument('--results', type=str, default="../results", help="path to results folder")
    parser.add_argument('--workspace', default="dog_results", type=str, help="path to results trial folder")
    parser.add_argument('--latest', default='ep0100', type=str, help="which epoch result you want to use for image path")
    parser.add_argument('--mode', default='rgb', type=str, help="mode of result, color(rgb) or textureless()")
    parser.add_argument('--clip', default="clip-ViT-B-32", type=str, help="CLIP model to encode the img and prompt")
    parser.add_argument('--all',  action='store_true', help="loop over all images in results folder")

    opt = parser.parse_args()

    #Load CLIP model
    model = SentenceTransformer(f'{opt.clip}')

    if opt.all:
        total_similarity = 0.
        total_items = 0
        
        for item in os.listdir(opt.results): 
            text = extract_text(item)
            if text == "skip":
                continue
            total_items += 1
            full_path = extract_full_path(opt, item)

            #Encode an image:
            img_emb = model.encode(Image.open(full_path))
        
            #Encode text descriptions
            text_emb = model.encode([f'{text}'])
        
            #Compute cosine similarities
            cos_scores = util.cos_sim(img_emb, text_emb)
            print("The final CLIP R-Precision is:", cos_scores[0][0].cpu().numpy())
            total_similarity += cos_scores
        total_similarity /= len(os.listdir(opt.results))
    else:
        #Encode an image:
        img_emb = model.encode(Image.open(f'{opt.results}/{opt.workspace}/validation/df_{opt.latest}_0005_{opt.mode}.png'))
    
        #Encode text descriptions
        text_emb = model.encode([f'{opt.text}'])
    
        #Compute cosine similarities
        cos_scores = util.cos_sim(img_emb, text_emb)
        print("The final CLIP R-Precision is:", cos_scores[0][0].cpu().numpy())

