import sys
sys.path.append("..")
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import argparse
import sys
from AudioToken.modules.BEATs.BEATs import BEATs, BEATsConfig
from AudioToken.modules.AudioToken.embedder import FGAEmbedder
import torch
from diffusers import StableDiffusionPipeline
import torchaudio


def aud_proc_beats(audio_path, rand_sec=0, input_length=10):
    """This function process the audio path and return audio sampling"""
    wav, sr = torchaudio.load(audio_path)
    wav = torch.tile(wav, (1, input_length))
    wav = wav[:, :sr*input_length]
    start = rand_sec * sr
    end = (rand_sec + input_length) * sr
    wav = wav[:, start:end]
    return wav[0].unsqueeze(0)
    
def get_text_embeds(device, prompt, audio_path, placeholder_token="<*>"):
       """This function tokenize and embedd the prompt, 
       given audio prompt we project the signal into textual prompt 
       and replace in placeholder_token"""
       model_key = "CompVis/stable-diffusion-v1-4"
       precision_t = torch.float32
       input_size = 768 * 3
       pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=precision_t)
       # Load Encoder audio network /phi(x)
       aud_checkpoint = torch.load(
           '../AudioToken/models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')
       aud_encoder = BEATs(BEATsConfig(aud_checkpoint['cfg']))
       
       embedder = FGAEmbedder(input_size=input_size, output_size=768)
       # Add the placeholder token in tokenizer
       num_added_tokens = pipe.tokenizer.add_tokens(placeholder_token)
       placeholder_token_id = pipe.tokenizer.convert_tokens_to_ids(placeholder_token)
       # Resize the token embeddings as we are adding new special tokens to the tokenizer
       pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
       # Read and process audio file
       audio_values = aud_proc_beats(audio_path)
       # Audio's feature extraction BETs
       aud_features = aud_encoder.extract_features(audio_values)[1]
       # Project Audio embedding to textual FGAEmbeeder
       audio_token = embedder(aud_features)
       # Replace empty token <*> with audio token 
       token_embeds = pipe.text_encoder.get_input_embeddings().weight.data
       token_embeds[placeholder_token_id] = audio_token.clone()

       inputs = pipe.tokenizer(prompt, padding='max_length', max_length=pipe.tokenizer.model_max_length, return_tensors='pt')
       embeddings = pipe.text_encoder(inputs.input_ids.to(device))[0]

       return embeddings
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="AudioDream", type=str, help="The model to test. "
                        "options: \"AudioDream\", \"DreamFusion\"")
    parser.add_argument('--audio', type=str, default="../audio_files/dog.wav", help="path to audio file")  
    parser.add_argument('--results', type=str, default="../dog_results", help="path to results folder")
    parser.add_argument('--text', default="A DSLR photo of a dog", type=str, help="text prompt")
    parser.add_argument('--latest', default='ep0100', type=str, help="which epoch result you want to use for image path")
    parser.add_argument('--mode', default='rgb', type=str, help="mode of result, color(rgb) or textureless()")
    parser.add_argument('--clip', default="clip-ViT-B-32", type=str, help="CLIP model to encode the img and prompt")

    opt = parser.parse_args()

    # Set device
    if torch.cuda.is_available():
      device = torch.device('cuda')
      print(f"Using CUDA device: {device}")
    else:
      device = torch.device('cpu')
      print("Using CPU")
    #Load CLIP model
    model = SentenceTransformer(f'{opt.clip}')

    #Encode an image:
    img_emb = model.encode(Image.open(f'{opt.results}/df_{opt.latest}_0008_{opt.mode}.png'))

    #Encode text descriptions
    if opt.model == "DreamFusion":
        text_emb = model.encode([f'{opt.text}'])
    else: # "AudioDream"
        opt.text = "A DSLR photo of a <*>"
        text_emb = get_text_embeds(device, [opt.text], opt.audio)

    #Compute cosine similarities
    cos_scores = util.cos_sim(img_emb, text_emb)
    print("The final CLIP R-Precision is:", cos_scores[0][0].cpu().numpy())

