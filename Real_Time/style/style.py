import sys
import os
import pdb
sys.path.insert(0, 'src')
import numpy as np
import scipy.misc
from scipy.optimize import optimize
from argparse import ArgumentParser
# from utils
# import evaluate

Content_weight = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2

LR = 1e-3
EPOCHS = 2
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_ITERATIONS = 2000
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'data/train2014'

