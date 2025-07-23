import os
import sys
import time
import yaml
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PIL import Image 
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset


def main(args):
    config = args.config
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training CA-Cut Model')
    parser.add_argument('--config', type=str, required=True, help='Location of the configuration file you would like to run')

    args = parser.parse_args()
    main(args)