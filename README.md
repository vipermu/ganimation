# GANimation

This project contains an implementation of [GANimation](https://arxiv.org/pdf/1807.09251.pdf) by Pumarola et al. using [StarGAN code](https://github.com/yunjey/stargan) by @yunjey as baseline.

We provide pretrained models and the preprocessed CelebA dataset to facilitate the training process of this model. We also explain though the process of preparing the data for training these models and the parameters that one needs to modify.

## Setup

#### Conda environment
Create your conda environment by just running the following command:
`conda env create -f environment.yml`

## Datasets

#### CelebA preprocessed dataset
Download and unzip the *CelebA* preprocessed dataset uploaded in [this link](https://www.dropbox.com/sh/mx3g9tggzl1kcd1/AAAueOQPKv3i9OJHRVCQEGcZa?dl=0) extracted from [MMLAB](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Here you can find a folder containing the aligned and resized 128x128 images as well as a _txt_ file containing their respective Action Units vectors computed using [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace). By default, this code assumes that you have these two elements in _`./data/celeba/`_.

#### Use your own dataset
If you want to use other datasets you will need to detect and crop bounding boxes around the face of each image, compute their corresponding Action Unit vectors and to resize them to 128x128px.

You can perform all these steps using [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace). First you will need to setup the project. They provide guides for [linux](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation) and [windows](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation). Once the models are compiled, read their [Action Unit wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units) and their [documentation](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments) on these models to know which is the command that you need to execute.

In my case the command was the following: `./build/bin/FaceLandmarkImg -fdir datasets/my-dataset/ -out_dir processed/my-processed-dataset/ -aus -simalign -au_static -nobadaligned -simsize 128 -format_aligned jpg -nomask`

After computing these Action Units, depending on the command that you have used, you will obtain different output formats. With the command that I used, I obtained a _csv_ file for each image containing its corresponding Action Units vector among extra information, a folder for each image containging the resized and cropped image and a _txt_ file with extra details about each image. Here you can find the code I used to create the _txt_ file having this output structure:

```python
import glob

output_txt = 'list_attr_mydataset.txt'

for idx, f in enumerate(glob.glob('./my-processed-dataset/*.csv')):
    with open(f, 'r') as csv_file:
        csv_file.readline()
        csv_list = csv_file.readline().split(', ')
        if float(csv_list[1]) >= 0.88:
            aus = " ".join(csv_list[2:19])
            open(output_txt, 'a').write(f.split('/')[-1].split('.')[0] + '.jpg ' + aus + '\n')
            
```
I also provide the code that I used to copy all the processed images into a folder:

```python
import os
import shutil

output_dir = './images'

os.mkdir(output_dir)

for root, dirs, files in os.walk('./my-processed-dataset'):
    for file in files:
        if 'jpg' in file:
            img_name = root.split('/')[-1].split('_')[0] + '.jpg'
            shutil.copy2(os.path.join(root, file), os.path.join(output_dir, img_name))
```

After having the Action Unit _txt_ file and the image folder you can move them to the directory of this project. By default, this code assumes that you have these two elements in _`./data/celeba/`_.

## Train the model

#### Parameters

Execute the main.py script. Check ut the default options with which it's running, you can change this parameters from the main.py script or from the command line as arguments.


## TODOs

- Clean Test function
- Add multi-gpu support
