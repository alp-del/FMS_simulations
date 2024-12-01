import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm
import pickle
import random
import resnet
import csv
from utils import get_datasets, get_model



def get_model_param_vec(model):
    """
    Return model parameters as a vector
    """
    vec = []
    for name,param in model.named_parameters():
        vec.append(param.detach().cpu().numpy().reshape(-1))
    return np.concatenate(vec, 0)

parser = argparse.ArgumentParser(description='Regular training and sampling for DLDR')
parser.add_argument('--n_components', default=40, type=int, metavar='N',
                    help='n_components for PCA') 
parser.add_argument('--params_start', default=0, type=int, metavar='N',
                    help='which epoch start for PCA') 
parser.add_argument('--params_end', default=81, type=int, metavar='N',
                    help='which epoch end for PCA') 
parser.add_argument('--datasets', metavar='DATASETS', default='CIFAR10', type=str,
                    help='The training datasets')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    help='model architecture (default: resnet32)')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_labelnoise0.1_resnet20', type=str)
args = parser.parse_args()

model = torch.nn.DataParallel(get_model(args))
model.cuda()

W = []
for i in range(args.params_start, args.params_end):
        ############################################################################
        # if i % 2 != 0: continue

    model.load_state_dict(torch.load(os.path.join(args.save_dir,  str(i) +  '.pt')))
    W.append(get_model_param_vec(model))
    
W = np.array(W)
print ('W:', W.shape)

W_hat = W@W.T
print('W_hat:',W_hat.shape)

np.savetxt('Wlabelnoise0.1.csv', W_hat, delimiter=',')