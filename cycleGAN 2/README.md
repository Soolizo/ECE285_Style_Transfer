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
