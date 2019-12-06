import os
import time
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data as td 
import torchvision as tv
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt 
import nntools as nt
from itertools import chain
from DnCNN import DnCNN

from styleDataSet import StyleTransDataset
from ganNet import Generator, Discriminator, CGANTrainer, CGANexp

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
print(device)


content_root_dir = "//datasets/ee285f-public/flickr_landscape/"
style_root_dir = "/datasets/ee285f-public/wikiart/wikiart/"
train_set = StyleTransDataset(content_root_dir, style_root_dir, "city", "Art_Nouveau_Modern")

transform = tv.transforms.Compose([
            tv.transforms.Resize((150, 150)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])

style_ref = Image.open("./starry_night.jpg").convert('RGB')
style_ref = transform(style_ref)
content_ref = Image.open("./house.jpg").convert('RGB')
content_ref = transform(content_ref)

gen2s = Generator(6).to(device)
gen2c = Generator(6).to(device)
dis_c = Discriminator(6).to(device)
dis_s = Discriminator(6).to(device)

cycleGan_trainer = CGANTrainer(gen2s, gen2c, dis_c, dis_s, device)
cycleGAN_exp = CGANexp(cycleGan_trainer, train_set, output_dir="cycleGAN_ckpt_long", batch_size = 2, picNum = 500, perform_validation_during_training=True)

cycleGAN_exp.run(num_epochs=200)