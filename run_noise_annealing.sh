#noise annealing
python main.py --audio "audio_files/dog.wav" --input_length 10 --workspace "results/noise_ann/trial_audioToken_noise_ann_1e-2_dog" --hf_key CompVis/stable-diffusion-v1-4 -O --noise_annealing 11 --iters 20000;
python main.py --audio "audio_files/dog.wav" --input_length 10 --workspace "results/noise_ann/trial_audioToken_noise_ann_1e-2_dog" --hf_key CompVis/stable-diffusion-v1-4 -O --noise_annealing 1e-1 --iters 20000;
python main.py --audio "audio_files/dog.wav" --input_length 10 --workspace "results/noise_ann/trial_audioToken_noise_ann_1e-2_dog" --hf_key CompVis/stable-diffusion-v1-4 -O --noise_annealing 1e-2 --iters 20000;
python main.py --audio "audio_files/dog.wav" --input_length 10 --workspace "results/noise_ann/trial_audioToken_noise_ann_1e-3_dog" --hf_key CompVis/stable-diffusion-v1-4 -O --noise_annealing 1e-3 --iters 20000;
python main.py --audio "audio_files/dog.wav" --input_length 10 --workspace "results/noise_ann/trial_audioToken_noise_ann_1e-4_dog" --hf_key CompVis/stable-diffusion-v1-4 -O --noise_annealing 1e-4 --iters 20000;
python main.py --audio "audio_files/dog.wav" --input_length 10 --workspace "results/noise_ann/trial_audioToken_noise_ann_1e-5_dog" --hf_key CompVis/stable-diffusion-v1-4 -O --noise_annealing 1e-5 --iters 20000;


