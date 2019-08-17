# GANimation

This project contains an implementation of [GANimation](https://arxiv.org/pdf/1807.09251.pdf) by Pumarola et al. using StarGAN [code](https://github.com/yunjey/stargan) as baseline.

We provide a pretrained model for the Generator and the preprocessed dataset


## Setup

#### Preprocessed dataset download
Download and unzip the *CelebA* resized images uploaded in [this link](https://www.dropbox.com/sh/mx3g9tggzl1kcd1/AAAueOQPKv3i9OJHRVCQEGcZa?dl=0) extracted from [MMLAB](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Here you can find the aligned and resized to 128x128 and a txt file with their corresponding Action Units. Save the *images_aligned* folder and the *txt* file into a folder called *data* indide this project. The Action Units where obtained from the OpenFace project (https://github.com/cmusatyalab/openface) .

#### Conda environment setup
Create your conda environment by just running the following command:
<conda env create -f environment.yml>

## Train the model

#### Parameters

Execute the main.py script. Check ut the default options with which it's running, you can change this parameters from the main.py script or from the command line as arguments.


## TODOs

- Clean Test function
- Add multi-gpu support
