# GANimation

This is an simplified implementation of the GANimation project (https://github.com/albertpumarola/GANimation) using StarGAN code (https://github.com/yunjey/stargan) as baseline.

To test this project follow this steps:
  1. Download and unzip the *CelebA* resized images uploaded in [this link](https://www.dropbox.com/sh/mx3g9tggzl1kcd1/AAAueOQPKv3i9OJHRVCQEGcZa?dl=0) extracted from [MMLAB](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Here you can find the aligned and resized to 128x128 and a txt file with their corresponding Action Units. Save the *images_aligned* folder and the *txt* file into a folder called *data* indide this project. The Action Units where obtained from the OpenFace project (https://github.com/cmusatyalab/openface) .
  2. Create a new virtual environment (python3 -m venv venv) and install all the required dependencies (pip install -r requirements.txt or just "pip install tensorflow torch torchvision")
  3. Execute the main.py script. Check ut the default options with which it's running, you can change this parameters from the main.py script or from the command line as arguments.
