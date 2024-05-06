# A23D - Audio to 3D Generative using Score Distillation Sampling and AudioToken

## Description

A pytorch implementation of the Audio-to-3D model, In
recent works, researchers have developed techniques to transfer pre-trained 2D
image to text diffusion models into 3D object synthesis models [*Poole et al. 2022*](https://dreamfusion3d.github.io/), [*Jiaxiang Tang et al. 2024*](https://dreamgaussian.github.io/) , without any 3D data. This project leverage the DreamFusion architecture and AudioToken paper [*Yariv et al. 2023*](https://pages.cs.huji.ac.il/adiyoss-lab/AudioToken/) to guidance 3d model such NeRF using audio modiality .

#put images results

## Table of Contents (Optional)
- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)
- [License](#license)

## Installation
```bash
# clone the repository
git clone https://github.com/arielkantorovich/Audio-to-3D.git

# Recommended venv ENV
conda create -n ENVNAME python=3.8

# Install basic requirments
pip install -r requirements.txt

# install expansion and correspond pytorch for CUDA rendering
1. conda install -c conda-forge cudatoolkit-dev
2. conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
3. pip install ./raymarching
4. pip install ./shencoder
5. pip install ./freqencoder
6. pip install ./gridencoder
```

**Tested environments:** WSL2 with Ubuntu 22.04, Python 3.8, CUDA 11.8, NVIDIA RTX 4090.

## Usage
First time running will take some time to compile the CUDA extensions.
Just change the aduio flag to your audio file path, --input_length is the audio length in sec for example audio dog.wav is 10 sec put 10.
```bash
python main.py --audio "audio_files/dog.wav" --input_length 10 --workspace trial_audioToken_dog --hf_key CompVis/stable-diffusion-v1-4 -O
```
## Acknowledgement
This work is based on an increasing list of amazing research works and open-source projects, thanks a lot to all the authors for sharing!

* [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion/tree/main)

* [DreamFusion: Text-to-3D using 2D Diffusion](https://dreamfusion3d.github.io/)
    ```
    @article{poole2022dreamfusion,
        author = {Poole, Ben and Jain, Ajay and Barron, Jonathan T. and Mildenhall, Ben},
        title = {DreamFusion: Text-to-3D using 2D Diffusion},
        journal = {arXiv},
        year = {2022},
    }
    ```

* [AudioToken: Adaptation of Text-Conditioned Diffusion Models for Audio-to-Image Generation](https://dreamfusion3d.github.io/)
    ```
    @article{yariv2023audiotoken,
  title={Audiotoken: Adaptation of text-conditioned diffusion models for audio-to-image generation},
  author={Yariv, Guy and Gat, Itai and Wolf, Lior and Adi, Yossi and Schwartz, Idan},
  journal={arXiv preprint arXiv:2305.13050},
  year={2023}
}
    ```