# GANimation 

This repo contains an implementation of [GANimation](https://arxiv.org/pdf/1807.09251.pdf) by Pumarola et al. based on [StarGAN code](https://github.com/yunjey/stargan) by @yunjey.

[Pretrained models](https://www.dropbox.com/sh/108g19dk3gt1l7l/AAB4OJHHrMHlBDbNK8aFQVZSa?dl=0) and the [preprocessed CelebA dataset](https://www.dropbox.com/s/payjdk08292csra/celeba.zip?dl=0) are provided to facilitate the use of this model as well as the process for preparing other datasets for training this model.

<p align="center">
  <img width="170" height="170" src="https://github.com/vipermu/ganimation/blob/master/video_results/frida.gif">
</p>

<p align="center">
  <img width="600" height="150" src="https://github.com/vipermu/ganimation/blob/master/video_results/eric_andre.gif">
</p>


## Setup

#### Conda environment
Create your conda environment by just running the following command:
`conda env create -f environment.yml`


## Datasets

#### CelebA preprocessed dataset
Download and unzip the *CelebA* preprocessed dataset uploaded in [this link](https://www.dropbox.com/s/payjdk08292csra/celeba.zip?dl=0) extracted from [MMLAB](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Here you can find a folder containing the aligned and resized 128x128 images as well as a _txt_ file containing their respective Action Units vectors computed using [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace). By default, this code assumes that you have these two elements in _`./data/celeba/`_.

#### Use your own dataset
If you want to use other datasets you will need to detect and crop bounding boxes around the face of each image, compute their corresponding Action Unit vectors and to resize them to 128x128px.

You can perform all these steps using [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace). First you will need to setup the project. They provide guides for [linux](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation) and [windows](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation). Once the models are compiled, read their [Action Unit wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units) and their [documentation](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments) on these models to know which is the command that you need to execute.

In my case the command was the following: `./build/bin/FaceLandmarkImg -fdir datasets/my-dataset/ -out_dir processed/my-processed-dataset/ -aus -simalign -au_static -nobadaligned -simsize 128 -format_aligned jpg -nomask`

After computing these Action Units, depending on the command that you have used, you will obtain different output formats. With the command that I used, I obtained a _csv_ file for each image containing its corresponding Action Units vector among extra information, a folder for each image containging the resized and cropped image and a _txt_ file with extra details about each image. You can find in _openface_utils_ folder the code that I used to extract all the Action Unit information in a _txt_ file and to group all the images into a single folder.

After having the Action Unit _txt_ file and the image folder you can move them to the directory of this project. By default, this code assumes that you have these two elements in _`./data/celeba/`_.

## Test the model
The pretrained models can be downloaded from [this](https://www.dropbox.com/sh/108g19dk3gt1l7l/AAB4OJHHrMHlBDbNK8aFQVZSa?dl=0) link. This folder contain the weights of both models (the Generator and the Discriminator) after training the model for 37 epochs.

By running `python main.py --mode test` the default test will be executed. This was the code used to generate the second _gif_ presented in this repo. It takes the Action Units extracted from a series of images and uses them as conditions on other faces. In the output can be seen how all the faces vary their expression in the same way.

#### Parameters
- *test_images_dir*: path to the reference test images.
- *test_attributes_path*: path to the _txt_ containing the Action Unit vectors of each image.
- *test_models_dir*: path to the pretrained models.
- *test_results_dir*: dir where to store the output images.

The sake of this test is to show an example on how this trained generator can be used so you can extract code to create your own tests. 

## Train the model

#### Parameters

You can either modify these parameters in `main.py` or by calling them as command line arguments.


##### Lambdas

- *lambda_cls*: classification lambda.
- *lambda_rec*: lambda for the cycle consistency loss.
- *lambda_gp*: gradient penalty lambda.
- *lambda_sat*: lambda for attention saturation loss.
- *lambda_smooth*: lambda for attention smoothing loss.

##### Training parameters

- *c_dim*: number of Action Units to use for training the model.
- *batch_size*
- *num_epochs*
- *num_epochs_decay*: number of epochs for start decaying the learning rate.
- *g_lr*: generator's learning rate.
- *d_lr*: discriminator's learning rate.

##### Pretrained models parameters
The weights are stored in the following format: `<epoch>-<iteration>-<G/D>.ckpt` where G and D represents the Generator and the Discriminator respectively. We save the optimizers's state in the same format and extension but adding '_optim'.

- *resume_iters*: iteration numbre from which we want to start the training. Note that we will need have a saved model corresponding to that exact iteration number.
- *first_epoch*: initial epoch for when we train from pretrained models.

##### Miscellaneous:
- *mode*: train/test.
- *image_dir*: path to your image folder.
- *attr_path*: path to your attributes _txt_ folder.
- *outputs_dir*: name for the output folder.

#### Virtual
- *virtual*: this flag activates the use of _cycle consistency loss_ during the training.

## Virtual Cycle Consistency Loss
The aim of this new component is to minimize the noise produced by the Action Unit regression. This idea was extracted from [Label-Noise Robust Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1905.02185) by Kaneko et al.. It is not proven that this new component improve the outcomes of the model but the masks seem to be darker when it is applied without lossing realism on the output images.

## TODOs

- Clean Test function.
- Add selective Action Units option for training.
- Add multi-gpu support.
- Smoother video generation.
