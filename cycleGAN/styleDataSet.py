import os
import time
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
import nntools as nt
from itertools import chain
from cGAN_model.DnCNN import DnCNN

class StyleGroupDataset(td.Dataset):
    def __init__(self, content_root_dir, style_root_dir,
                 content_categories, artist = 22, mode="train", trainCat = "Artist", 
                 image_size=(150, 150), sigma=30):
        assert type(content_categories) == list, "input content_categories must be a list"
        assert 0 <= artist <= 22, "invalid artist"
        
        super(StyleGroupDataset, self).__init__()
        
        self.image_size = image_size
        self.sigma = sigma
        self.mode = mode
        self.content_cat = content_categories
        self.content_files = []
        
        for content_category in content_categories:
            content_dir = os.path.join(content_root_dir, content_category) 
            self.content_files += [os.path.join(content_dir, f) for f in os.listdir(content_dir)]
        
        self.artist = artist
        self.style_files = pd.read_csv(os.path.join(style_root_dir, f"{trainCat}/{trainCat.lower()}_{mode}" ))
        self.style_files = self.style_files.loc[self.style_files.iloc[:,2] == self.artist]   # vincent-van-gogh
        self.images_dir = os.path.join(style_root_dir,'wikiart')
        
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize(self.image_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])

    def __len__(self):
        return len(self.style_files)
    
    def content_num(self):
        return len(self.content_files)
    
    def style_num(self):
        return len(self.style_files)

    def __repr__(self):
        return f"Dataset(mode={self.mode}, image_size={self.image_size}, artist={self.artist}, content={self.content_cat})"
    
    def __getitem__(self, idx):
        content_idx = idx % self.content_num()
        style_idx = idx % self.style_num()
        
        content = Image.open(self.content_files[idx]).convert('RGB')
        
        style_path = os.path.join(self.images_dir, self.style_files.iloc[style_idx][0])
        style = Image.open(style_path).convert('RGB')

        content = self.transform(content)
        style = self.transform(style)
        return content, style
    
class StyleTransDataset(td.Dataset):

    def __init__(self, content_root_dir, style_root_dir,
                 content_category, style_category,
                 image_size=(150, 150), sigma=30): 
        super(StyleTransDataset, self).__init__()
        self.content_cat = content_category
        self.style_cat = style_category
        self.image_size = image_size
        self.sigma = sigma
        
        self.content_dir = os.path.join(content_root_dir, content_category) 
        self.style_dir = os.path.join(style_root_dir, style_category) 
        self.content_files = os.listdir(self.content_dir)
        self.style_files = os.listdir(self.style_dir)
        
    def __len__(self):
        return len(self.content_files) #min(self.content_num(), self.style_num())
    
    def content_num(self):
        return len(self.content_files)
    
    def style_num(self):
        return len(self.style_files)
    
    def __repr__(self):
        return (f"StyleTransDataset(category: {self.content_cat}"
                f", image_size={self.image_size}, sigma={self.sigma})")

    def __getitem__(self, idx):
        content_path = os.path.join(self.content_dir, self.content_files[idx]) 
        content = Image.open(content_path).convert('RGB')
        
#         style_path = os.path.join(self.style_dir, self.style_files[idx]) 
#         style = Image.open(style_path).convert('RGB')
        
        return self.trans(content) #, self.trans(style)
    
    def trans(self, img):
        i = np.random.randint(img.size[0] - self.image_size[0]) 
        j = np.random.randint(img.size[1] - self.image_size[1]) 

        img = img.crop([i, j , i + self.image_size[0], j + 
                            self.image_size[1]]) 
        
        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        
        img = transform(img)
        return img
    
class ContentTestDataset(td.Dataset):

    def __init__(self, content_root_dir, content_category, 
                 image_size=(150, 150), sigma=30): 
        super(ContentTestDataset, self).__init__()
        self.content_cat = content_category
        self.image_size = image_size
        self.sigma = sigma
        
        self.content_dir = os.path.join(content_root_dir, content_category) 
        self.content_files = os.listdir(self.content_dir)
        
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize(self.image_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        
    def __len__(self):
        return len(self.content_files) 
    
    def content_num(self):
        return len(self.content_files)
    
    def __repr__(self):
        return (f"StyleTransDataset(category: {self.content_cat}"
                f", image_size={self.image_size}, sigma={self.sigma})")

    def __getitem__(self, idx):
        content_path = os.path.join(self.content_dir, self.content_files[idx]) 
        content = Image.open(content_path).convert('RGB')
          
        return self.transform(content)