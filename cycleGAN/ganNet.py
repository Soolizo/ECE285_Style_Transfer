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
from cGAN_model.DnCNN import DnCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = tv.transforms.Compose([
            tv.transforms.Resize((150, 150)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])

style_ref = Image.open("./starry_night.jpg").convert('RGB')
style_ref = transform(style_ref)


# Generator and Discriminator for GAN
# Generator use DnCNN from HW4, with MSE loss

class Generator(DnCNN):
    def __init__(self, D, C=64):
        super(Generator, self).__init__(D)
        
    def criterion(self, y, d):
        return nn.MSELoss()(y, d)

    
# Discriminator use 4 layers of CNN (3 -> 16, 64, 256, 256) along with instNorm and RELU for transforming the image. Then, it use CNN(256 -> 1) for discriminating decision making

class Discriminator(DnCNN):
    def __init__(self, D, C=64):
        super(Discriminator, self).__init__(D)
        
        self.cnn = nn.Sequential()
        dims = [3, 16, 64, 256, 256]
        for i in range(len(dims) - 1):
            self.cnn.add_module(f"conv2d{i}", nn.Conv2d(dims[i], dims[i + 1], 3, padding = 1))
            self.cnn.add_module(f"instNorm{i}", nn.InstanceNorm2d(dims[i + 1]))
            self.cnn.add_module(f"relu{i}", nn.LeakyReLU(0.2, True))
        
        self.cnn.add_module("cov2dFL", nn.Conv2d(256, 1, 3, padding=1, bias=False))
        
    def forward(self, x):
        h = self.cnn(x)
        h = h.view(h.size(0), -1)
        return h
    
    def criterion(self, y, d):
        return nn.L1Loss()(y, d)

# Trainer class used to train cycle GAN, geting loss/gradient for generator and discriminator and do optimization with Adam.    
    
    
class CGANTrainer():

    def __init__(self, device, D=6):
        self.device = device
        
        self.gen2s = Generator(D).to(device)
        self.gen2c = Generator(D).to(device)
        self.dis_c = Discriminator().to(device)
        self.dis_s = Discriminator().to(device)
        
        self.lr = 2e-3
        self.adam_gen = torch.optim.Adam(chain(self.gen2s.parameters(), self.gen2c.parameters()), 
                                         lr=self.lr, betas=(0.5,0.999))
        self.adam_dis_c = torch.optim.Adam(self.dis_c.parameters(), lr=self.lr, betas=(0.5,0.999))
        self.adam_dis_s = torch.optim.Adam(self.dis_s.parameters(), lr=self.lr, betas=(0.5,0.999))
        
        self.scheduler_gen = torch.optim.lr_scheduler.StepLR(self.adam_gen, step_size=2, gamma=0.99)
        self.scheduler_dis_c = torch.optim.lr_scheduler.StepLR(self.adam_dis_c, step_size=2, gamma=0.99)
        self.scheduler_dis_s = torch.optim.lr_scheduler.StepLR(self.adam_dis_s, step_size=2, gamma=0.99)
        
        self.l1Loss = nn.L1Loss().to(self.device)
        self.l2Loss = nn.MSELoss().to(self.device)
        
    def forward(self, content, style):
        """
        Prepare generated tensors for training.
        Must be called before calling train_generator/train_discriminator
        """
        self.content = content
        self.style = style
        self.S_c = self.gen2s(content)
        self.C_S_c = self.gen2c(self.S_c)
        self.C_s = self.gen2c(style)
        self.S_C_s = self.gen2s(self.C_s)
    def train_generator(self):  
        self.adam_gen.zero_grad()

        totalLoss = 0

        # get Discriminator Loss
        disS= self.dis_s(self.S_c)
        real_var = Variable(torch.cuda.FloatTensor(disS.shape).fill_(1.0),
                            requires_grad = False)
        totalLoss += self.l2Loss(disS, real_var)
        
        disC = self.dis_c(self.C_s)
        real_var = Variable(torch.cuda.FloatTensor(disC.shape).fill_(1.0),
                            requires_grad = False)
        totalLoss += self.l2Loss(disC, real_var)

        # get Cycle GAN Loss
        totalLoss += self.l1Loss(self.C_S_c, self.content)
        totalLoss += self.l1Loss(self.S_C_s, self.style)

        # update generator
        totalLoss.backward()
        self.adam_gen.step()
        return totalLoss
    
    def train_discriminator(self, mode):
        """
        Train the discriminator. 
        mode == 0: train the discriminator for style 
        mode == 1: train the discriminator for content
        """

        assert (mode == 0 or mode == 1), "input must be 0(train dis_s) or 1(train dis_c)" 

        if mode == 0:
            # Train the style discriminator           
            adam_dis = self.adam_dis_s
            dis, gen, ori = self.dis_s, self.S_c, self.style
        else:
            # Train the content discriminator
            adam_dis = self.adam_dis_c
            dis, gen, ori = self.dis_c, self.C_s, self.content
        
            
        adam_dis.zero_grad()   
        totalLoss = 0
        
        disReal = dis(ori)
        real_var = Variable(torch.cuda.FloatTensor(disReal.shape).fill_(1.0),
                            requires_grad = False)
        totalLoss += self.l2Loss(disReal, real_var)
        
        # get Discriminator Loss
        dis_fake = dis(gen.detach())
        fake_var = Variable(torch.cuda.FloatTensor(dis_fake.shape).fill_(0.0),
                            requires_grad = False)
        totalLoss += self.l2Loss(dis_fake, fake_var)
        
        # update discriminator
        totalLoss.backward()
        adam_dis.step()
        return totalLoss
    
    def train(self, content, style):
        dis_loss, gen_loss = 0, 0 
        
        self.forward(content, style)
        # train discrimiator
        dis_loss += self.train_discriminator(0).item()
        dis_loss += self.train_discriminator(1).item()
            
        # train generator
        gen_loss = self.train_generator().item()
        
        return gen_loss, dis_loss, dis_loss + gen_loss
    
    def update_lr(self):
        self.scheduler_gen.step()
        self.scheduler_dis_c.step()
        self.scheduler_dis_s.step()

#Experiment inherited from nt.Experiment. Being able to run experiment, save/load check points.        
        
class CGANexp(nt.Experiment):
    def __init__(self, cGANTrainer, train_set, output_dir, 
                 picNum = 100, batch_size=16, device = device,
                 perform_validation_during_training=False):  
        # Initialize
        self.history = []
        self.trainer = cGANTrainer
        self.device = device
        
        self.picNum = picNum
        self.net = self.trainer.gen2c

        self.test_loader = td.DataLoader(train_set,
                  batch_size=4, shuffle=False, 
                  drop_last=True, pin_memory=True)
                
        self.toRecover = {
            'contentGenNet': self.trainer.gen2c,
            'styleGenNet': self.trainer.gen2s,
            'contentDisNet': self.trainer.dis_c,
            'styleDisNet': self.trainer.dis_s,
            'genAdam': self.trainer.adam_gen,
            'contentDisAdam': self.trainer.adam_dis_c,
            'styleDisAdam': self.trainer.adam_dis_s,
            'history': self.history
           }
        
        # Define checkpoint paths
        if output_dir is None:
            output_dir = 'experiment_{}'.format(time.time())
        os.makedirs(output_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(output_dir, "checkpoint.pth.tar")
        self.config_path = os.path.join(output_dir, "config.txt")
        self.log_path = os.path.join(output_dir, "log.txt")

        # Transfer all local arguments/variables into attributes
        locs = {k: v for k, v in locals().items() if k is not 'self'}
        self.__dict__.update(locs)

        # Load checkpoint and check compatibility
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'r') as f:
                if f.read()[:-1] != repr(self):
                    raise ValueError(
                        "Cannot create this experiment: "
                        "I found a checkpoint conflicting with the current setting.")
            print("Done Load from Checkpoint!")
            self.load()
        else:
            self.save()
        
    def setting(self):
        """Returns the setting of the experiment."""
        return {'contentGenNet': self.trainer.gen2c,
                'TrainSet': self.train_set,
                'styleGenNet': self.trainer.gen2s,
                'contentDisNet': self.trainer.dis_c,
                'styleDisNet': self.trainer.dis_s,
                'genAdam': self.trainer.adam_gen,
                'contentDisAdam': self.trainer.adam_dis_c,
                'styleDisAdam': self.trainer.adam_dis_s,
                'TrainSet': self.train_set,
                'BatchSize': self.batch_size,
                'PerformValidationDuringTraining': self.perform_validation_during_training}

    def __repr__(self):
        """Pretty printer showing the setting of the experiment. This is what
        is displayed when doing ``print(experiment)``. This is also what is
        saved in the ``config.txt`` file.
        """
        string = ''
        for key, val in self.setting().items():
            string += '{}({})\n'.format(key, val)
        return string
    
    def state_dict(self):
        """Returns the current state of the experiment."""
        return {'contentGenNet': self.trainer.gen2c.state_dict(),
                'styleGenNet': self.trainer.gen2s.state_dict(),
                'contentDisNet': self.trainer.dis_c.state_dict(),
                'styleDisNet': self.trainer.dis_s.state_dict(),
                'genAdam': self.trainer.adam_gen.state_dict(),
                'contentDisAdam': self.trainer.adam_dis_c.state_dict(),
                'styleDisAdam': self.trainer.adam_dis_s.state_dict(),
                'history': self.history
               }
    
    
    def load_state_dict(self, checkpoint):
        """Loads the experiment from the input checkpoint."""
        for key, val in checkpoint.items():
            if key not in self.toRecover:
                raise AttributeError(f"Loading is Wrong! Key is {key}")
            if key == 'history':
                self.history = val
            else:
                self.toRecover[key].load_state_dict(val)
               
        nets = [self.trainer.gen2c, self.trainer.gen2s, 
                self.trainer.dis_c, self.trainer.dis_s]
        adams = [self.trainer.adam_gen, self.trainer.adam_gen,
                 self.trainer.adam_dis_c, self.trainer.adam_dis_s]
        
        for net, optimizer in zip(nets, adams):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
                        
    def load(self):
        """Loads the experiment from the last checkpoint saved on disk."""
        checkpoint = torch.load(self.checkpoint_path,
                                map_location=self.device)
        self.load_state_dict(checkpoint)
        del checkpoint
        
    def save(self):
        """Saves the experiment on disk, i.e, create/update the last checkpoint."""
        torch.save(self.state_dict(), self.checkpoint_path)
        with open(self.config_path, 'w') as f:
            print(self, file=f)
        
    def run(self, num_epochs, plot=None):
        global style_ref
        
        start_epoch = self.epoch
        print("Start/Continue training from epoch {}".format(start_epoch))
        if plot is not None:
            plot(self)
                
        for epoch in range(start_epoch, num_epochs):
            s = time.time()
            i = 0
            gen_loss = []
            dis_c_loss = []
            dis_s_loss = []
            total_loss = []
            
            for content in self.test_loader:
                if i > self.picNum:
                    break
                content = content.to(self.device)
                style = style_ref[np.newaxis].to(self.device)
                
                self.trainer.forward(content, style)
                
                gen_loss.append(self.trainer.train_generator().item())
                dis_c_loss.append(self.trainer.train_discriminator(0).item())
                dis_s_loss.append(self.trainer.train_discriminator(1).item())
            
                i += 1
                
            self.history.append((np.mean(gen_loss+dis_c_loss+dis_s_loss), np.mean(gen_loss), np.mean(dis_c_loss+dis_s_loss)))
            
            with open(self.log_path, 'w') as f:
                print("Epoch {} (Time: {:.2f}s)".format(
                    self.epoch, time.time() - s), file=f)
            self.save()
            if plot is not None:
                plot(self)