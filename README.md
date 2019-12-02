# ECE285_Style-Transfer


## Description
This is ECE 285 final project Style Transfer developed by team Learning Machine composed of Bowen Zhao, Bolin HE, Kunpeng Liu.



Origin Style Transfer
===========
This is the version based on Gatys paper Style Transfer, it has test on the UCSD DSMLP.
The code locates in Origin folder.

## Requirements

install the package as fllow:  
 $ pip3 install torch torchvision  
Besides this code need to download VGG19 network  

## Code organization

[demo.ipynb](https://github.com/Soolizo/ECE285_Style-Transfer/blob/master/Origin/Demo.ipynb) -- Run a demo of our code and showing different alpha/beta ratio's effect on the output image.  
[image.py](https://github.com/Soolizo/ECE285_Style-Transfer/blob/master/Origin/image.py) -- Contains the function that used to load and show image.  
[loss.py](https://github.com/Soolizo/ECE285_Style-Transfer/blob/master/Origin/loss.py) -- Contains the function to calculate the content loss, gram martix, style loss.  
[model.py](https://github.com/Soolizo/ECE285_Style-Transfer/blob/master/Origin/model.py) -- Contains the VGG19 network and record the layer that need to calculate content loss and style loss.  
[run.py](https://github.com/Soolizo/ECE285_Style-Transfer/blob/master/Origin/run.py) -- Contains the function with the initialization and how to compute the loss and optimize them.  



Real Time Transfer
===========
