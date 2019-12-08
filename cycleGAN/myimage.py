import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data as td 
import torchvision as tv
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt 
from itertools import chain

import styleDataSet as sd
import ganNet as gn

from cGAN_model.DnCNN import DnCNN
import cGAN_model.nntools as nt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
content_root_dir = "/datasets/ee285f-public/flickr_landscape/"
style_root_dir = "/datasets/ee285f-public/wikiart/"

train_set = sd.StyleGroupDataset(content_root_dir, style_root_dir,
                              content_categories=["city"])

long_train_set = sd.StyleGroupDataset(content_root_dir, style_root_dir,
                                  content_categories=["forest", "lake", "city"])


test_set = sd.ContentTestDataset(content_root_dir, "road")



def myimshow(image, ax=plt):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1]) 
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image) 
    ax.axis('off') 
    return h

def plot_basic(exp, fig, axes, content, style, visu_rate=2, save_path=None): 
    if exp.epoch % visu_rate != 0:
        return
    with torch.no_grad():
        transfered = exp.trainer.gen2s(content[np.newaxis].to(exp.trainer.gen2s.device))[0] 
    axes[0][0].clear()
    axes[0][1].clear()
    axes[1][0].clear()
    axes[1][1].clear()
    myimshow(content, ax=axes[0][0]) 
    axes[0][0].set_title('Content image') 
    
    myimshow(style, ax=axes[0][1]) 
    axes[0][1].set_title('Style image')

    myimshow(transfered, ax=axes[1][0]) 
    axes[1][0].set_title('Transfered image')
    
    axes[1][1].plot([exp.history[k][0].item() 
                     for k in range(exp.epoch)],label="Total Loss")
    axes[1][1].plot([exp.history[k][1].item() 
                     for k in range(exp.epoch)],label="Gen Loss")
    axes[1][1].plot([exp.history[k][2].item()
                     for k in range(exp.epoch)],label="Dis Loss")
    
    axes[1][1].legend(loc='best')
    
    axes[1][1].set_xlabel("Epoch")
    axes[1][1].set_ylabel("Loss")
    
    plt.tight_layout() 
    fig.canvas.draw()
    
    if save_path is not None:
        fig.savefig(save_path)

def plot_exp(exp, fig, axes, ind=0, save_path=None):
    if save_path is not None:
        save_path = "./proj_report_img/" + save_path
    
    plot_basic(exp, fig = fig,
               axes = axes,content = train_set[ind][0],
               style = train_set[ind][1],
               save_path = save_path)
    
def multi_final_result(save_path, use_train, exp, *test_ids):
    assert (save_path==None or type(save_path)==str), "first input must be string for save path" 
    assert (type(use_train)==bool), "second input must be True or False" 
    
    img_num = len(test_ids)
    fig, axes = plt.subplots(ncols=img_num, nrows=2) 
    
    for i in range(img_num):
        if use_train:
            to_test = long_train_set[test_ids[i]][0][np.newaxis].to(device)
        else:
            to_test = test_set[test_ids[i]][np.newaxis].to(device)
            
        myimshow(to_test[0], axes[0][i])
        
        with torch.no_grad():
            myimshow(exp.trainer.gen2s(to_test)[0] , ax=axes[1][i])
            
        axes[0][i].set_title("Original")
        axes[1][i].set_title("Transferred")
    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig("./proj_report_img/" + save_path)
