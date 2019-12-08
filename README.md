# ECE285 Style Transfer

This is UCSD ECE 285 Final Project B-Style Transfer, developed by team Learning Machine composed of 

`Bowen Zhao`

`Bolin He`

`Kunpeng Liu`.

  </br>
  </br>
  
## Part I: Neural Style Transfer

### Description

This is the version based on Gatys paper Style Transfer, it has tested on the UCSD DSMLP.
The code locates in [__`Origin`__](./Origin) folder. 

### Requirements

install the package as fllow:
```
 $ pip3 install torch torchvision  
```
The origin style transfer also requires to download VGG19 network. 

### Code organization

[`demo.ipynb`](https://github.com/Soolizo/ECE285_Style-Transfer/blob/master/Origin/Demo.ipynb) -- Run a demo of our code and showing different alpha/beta ratio's effect on the output image.   
[`image.py`](https://github.com/Soolizo/ECE285_Style-Transfer/blob/master/Origin/image.py) -- Contains the function that used to load and show image.  
[`loss.py`](https://github.com/Soolizo/ECE285_Style-Transfer/blob/master/Origin/loss.py) -- Contains the function to calculate the content loss, gram martix, style loss.  
[`model.py`](https://github.com/Soolizo/ECE285_Style-Transfer/blob/master/Origin/model.py) -- Contains the VGG19 network and record the layer that need to calculate content loss and style loss.  
[`run.py`](https://github.com/Soolizo/ECE285_Style-Transfer/blob/master/Origin/run.py) -- Contains the function with the initialization and how to compute the loss and optimize them.  
  

### Example
We use the content image from one of Picasso's masterpiece, and style image from Kanagawa. Here is the output with α/β ratio equals to 10<sup>8</sup>(α is the weight of style loss, β is the weight of content loss).

<div align=center />
<img src="Origin/image/Cubic.png" width = "256" height = "256" alt="Content" align=center />

<img src="Origin/image/Kanagawa.png" width = "256" height = "256" alt="Style" align=center />

<img src="Origin/image/result.png" width = "256" height = "256" alt="Result" align=center />
</div>
    
</br>
</br>
</br>
</br>
</br>
    	 
## Part II: Image-to-Image Translation using Cycle-GANs

### Description
Image-to-Image Translation using Cycle-GANs has been tested on UCSD DSMLP.
The code locates in [__`cycleGAN`__](./cycleGAN) folder.

In this part, we implemented a real-time Sytle Transfer using Cycle-GANs introduced in paper [_Unpaired Image-to-Image Translation
using Cycle-Consistent Adversarial Networks_](https://arxiv.org/pdf/1703.10593.pdf). 

In our experiment, the content train dataset is landcape pictures from [FLickr](https://www.flickr.com/groups/landcape/), and the style train dataset is Vincent van Gogh artworks from [WikiArt5](https://www.wikiart.org/en). We successful use DnCNN in Assignment 4 with three configuration: 

[`exp_smallBS`](./cycleGAN/proj_report_img/domain_exp_long_train.png): Experiment small batch-size without decay

[`exp_largeBS`](./cycleGAN/proj_report_img/domain_large_bs_train.png): Experiment large batch-size without decay

[`exp_largeBS_decay`](./cycleGAN/proj_report_img/domain_exp_largeBS_decay_train.png): Experiment large batch-size with decay

### Requirements

In our implementation of the project, we do not require any extra packages besides the ones provided by the class environment: `os`, `time`, `numpy`, `pandas`, `torch`, `torchvision`, `PIL`, and `matplotlib`.

To run the training scripts outside of the class environment, you may need to install some of the packages above. To install a missing package `LIBRARY_MISSING`, you may do the following command:  
```console
$ pip install --user LIBRARY_MISSING
```

### Code organization

`./`
 * [`proj_report_img/`](./cycleGAN/proj_report_img)
 * [`cGAN_model/`](./cycleGAN/cGAN_model)
   * `__init__.py`
   * [`ganNet.py`](./cycleGAN/cGAN_model/ganNet.py), [`ganNet_domain.py`](./cycleGAN/cGAN_model/ganNet_domain.py)
   
   			Main classes of Generator, Discriminator,
            Trainer, and Training Experiment for 
            picture-orientated CycleGAN and 
            domain-orientated CycleGAN respectively
   * [`styleDataSet.py`](./cycleGAN/cGAN_model/styleDataSet.py), [`domainStyleDataSet.py`](./cycleGAN/cGAN_model/domainStyleDataSet.py)
   
   			Class(_td.Dataset_) for loading dataset 
            from given path for picture-orientated CycleGAN and 
            domain-orientated CycleGAN respectively
            
    * [`nntools.py`](./cycleGAN/cGAN_model/nntools.py), [`DnCNN.py`](./cycleGAN/cGAN_model/DnCNN.py)
   			
            Classes adapeted from Assignment 4,
            used for building CycleGAN networks
 * [`cGAN_ckpts/`](./cycleGAN/cGAN_ckpts)
 
        Contains numerous checkpoint folders 
        for loading/continue pre-trained models

 * [`domain_CycleGAN_demo.ipynb`](./cycleGAN/domain_CycleGAN_demo.ipynb) 
 
		Run a demo of our code transfering content picture
        based on an aritist's domain. (reproduces 
        Figure 3 of our report)
 * [`cGAN_train.py`](./cycleGAN/cGAN_train.py)  
 
        Run training of our model through python script.
        Able to run in background with flags to customize settings 
        such as domain to train, use large train_set, and etc..  
 * [`myimage.py`](./cycleGAN/myimage.py)  
 
        Some function that help to plot. 
### Example
To visualize the result, check 
 * [`domain_CycleGAN_demo.ipynb`](./cycleGAN/domain_CycleGAN_demo.ipynb) 
 
 <div align=center />
<img src="cycleGAN/proj_report_img/domain_exp_largeBS_decay_train.png" height = "330"   />

<img src="cycleGAN/proj_report_img/domain_exp_largeBS_decay.png" height = "330" />
</div>
 
  
 
