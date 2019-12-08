# ECE285 Style ransfer


### Description
This is ECE 285 final project b Style Transfer developed by team Learning Machine composed of Bowen Zhao, Bolin HE, Kunpeng Liu.

### Requirements

install the package as fllow:
```
 $ pip3 install torch torchvision  
```
The origin style transfer also needs to download VGG19 network. And the training set of Cycle-GAN locates on the DSMLP of UCSD with the following directions:
```
content_root_dir = "//datasets/ee285f-public/flickr_landscape/"

style_root_dir = "/datasets/ee285f-public/wikiart/wikiart/"
```
  
  
  
Neural Style Transfer
===========
This is the version based on Gatys paper Style Transfer, it has tested on the UCSD DSMLP.
The code locates in Origin folder.

### Code organization

[demo.ipynb](https://github.com/Soolizo/ECE285_Style-Transfer/blob/master/Origin/Demo.ipynb) -- Run a demo of our code and showing different alpha/beta ratio's effect on the output image.  
[image.py](https://github.com/Soolizo/ECE285_Style-Transfer/blob/master/Origin/image.py) -- Contains the function that used to load and show image.  
[loss.py](https://github.com/Soolizo/ECE285_Style-Transfer/blob/master/Origin/loss.py) -- Contains the function to calculate the content loss, gram martix, style loss.  
[model.py](https://github.com/Soolizo/ECE285_Style-Transfer/blob/master/Origin/model.py) -- Contains the VGG19 network and record the layer that need to calculate content loss and style loss.  
[run.py](https://github.com/Soolizo/ECE285_Style-Transfer/blob/master/Origin/run.py) -- Contains the function with the initialization and how to compute the loss and optimize them.  
  

### Example
We use the content image from one of Picasso's masterpiece, and style image from Kanagawa. Here is the output with α/β ratio equals to 10<sup>8</sup>(α is the weight of style loss, β is the weight of content loss).

<div align=center />
<img src="Origin/image/Cubic.png" width = "256" height = "256" alt="Content" align=center />

<img src="Origin/image/Kanagawa.png" width = "256" height = "256" alt="Style" align=center />

<img src="Origin/image/result.png" width = "256" height = "256" alt="Result" align=center />
</div>

  
Image-to-Image Translation using Cycle-GANs
===========
Description
===========
This is the second part of project Style Transfer developed by team composed of Bolin He, Kunpeng Liu, Bowen Zhao.

In this part, we implemented a real-time Sytle Transfer using Cycle-GANs introduced in paper [_Unpaired Image-to-Image Translation
using Cycle-Consistent Adversarial Networks_](https://arxiv.org/pdf/1703.10593.pdf). In our pretrained model saved in checkpoint folders, we used the picture [_starry night_](https://www.wikiart.org/en/vincent-van-gogh/the-starry-night-1889) as our style referrence and trained our model with landscape images from [FLickr](https://www.flickr.com/groups/landcape/) as contents to be transferred.

Requirements
============
In our implementation of the project, we do not require any extra packages besides the ones provided by the class environment: `os`, `time`, `numpy`, `pandas`, `torch`, `torchvision`, `PIL`, and `matplotlib`.

To run the training scripts outside of the class environment, you may need to install some of the packages above. To install a missing package `LIBRARY_MISSING`, you may do the following command:  
```console
$ pip install --user LIBRARY_MISSING
```

Code organization
=================  
`./`
 * [`proj_report_img/`](./proj_report_img)
 * [`cGAN_model/`](./cGAN_model)
   * `__init__.py`
   * [`ganNet.py`](./cGAN_model/ganNet.py), [`ganNet_domain.py`](./cGAN_model/ganNet_domain.py)
   
   			Main classes of Generator, Discriminator,
            Trainer, and Training Experiment for 
            picture-orientated CycleGAN and 
            domain-orientated CycleGAN respectively
   * [`styleDataSet.py`](./cGAN_model/styleDataSet.py), [`domainStyleDataSet.py`](./cGAN_model/domainStyleDataSet.py)
   
   			Class(_td.Dataset_) for loading dataset 
            from given path for picture-orientated CycleGAN and 
            domain-orientated CycleGAN respectively
            
    * [`nntools.py`](./cGAN_model/nntools.py), [`DnCNN.py`](./cGAN_model/DnCNN.py)
   			
            Classes adapeted from Assignment 4,
            used for building CycleGAN networks
 * [`cGAN_ckpts/`](./cGAN_ckpts)
 
        Contains numerous checkpoint folders 
        for loading/continue pre-trained models
 * [`CycleGAN_demo.ipynb`](./CycleGAN_demo.ipynb)  
 
        Run a demo of our code transfering content picture 
        based on one single style picture. (reproduces
        Figure 1 and Figure 2 of our report)  
 * [`domain_CycleGAN_demo.ipynb`](./domain_CycleGAN_demo.ipynb) 
 
		Run a demo of our code transfering content picture
        based on an aritist's domain. (reproduces 
        Figure 3 of our report)
 * [`cGAN_train.py`](./cGAN_train.py)  
 
        Run training of our model through python script.
        Able to run in background with flags to customize settings 
        such as domain to train, use large train_set, and etc..  
 * [house.jpg](./house.jpg), [starry_night.jpg](./starry_night.jpg)

