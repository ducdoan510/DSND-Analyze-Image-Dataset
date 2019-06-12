# DSND-Analyze-Image-Dataset

## Overview
This project classifies the image dataset with transfer learning method using PyTorch. On second part of the project, a simple CLI application is implemented to allow user to classify own images

## Instruction
Under _Part 2_ folder:
1. Run _train.py_ file to create a checkpoint for a customized model
<pre><code>
usage: train.py [-h] [--save_dir SAVE_DIR] [--arch ARCH]
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--epochs EPOCHS] [--gpu]
                data_directory

Training image classifier

positional arguments:
  data_directory        image data directory path with train/valid/test
                        subfolders

optional arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Model checkpoint saving
  --arch ARCH           Pretrained model from torchvision
  --learning_rate LEARNING_RATE
                        Optimizer learning rate
  --hidden_units HIDDEN_UNITS
                        Number of hidden units in customized classifier
  --epochs EPOCHS       Number of training epochs
  --gpu                 Flag to set using GPU

</code></pre>

2. Run _predict.py_ to use the model checkpoint to predict
<pre><code>
usage: predict.py [-h] [--top_k TOP_K] [--category_name CATEGORY_NAME] [--gpu]
                  image_path checkpoint

Predict image type

positional arguments:
  image_path            path to image to predict
  checkpoint            path to model checkpoint

optional arguments:
  -h, --help            show this help message and exit
  --top_k TOP_K         Number of top classes to display
  --category_name CATEGORY_NAME
                        path to json file representing mapping flower category
                        to actual name
  --gpu                 Flag to set using GPU

</code></pre>


## File structure
1. Part 1: contains the Jupiter notebook that preprocesses the image data and train the classifier using transfer learning with PyTorch
2. Part 2: a Python module extracting the code from the notebook
  - image.py: utility functions that process image data
  - model.py: utility functions that train the image classifier
  - train.py: entry point for CLI application for creating the model checkpoint for customized classifier
  - predict.py: entry point for CLI application that uses the checkpoint to load the model to predict image class
 
## Acknowledgement
This project is a part of Udacity Data Science Nanodegree
