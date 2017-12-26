# Fast Style Transfer

Implementation of [Instance Normalization](https://arxiv.org/abs/1607.08022) and a generator network for one shot stylization. 


## Demonstration
 
The below gif is the result of the processing done by using a video window on screen and using the ImageGrab feature. OpenCV is having some issues on my new laptop and thus I have no committed a Webcam version yet. 

![Alt text](https://github.com/akhauriyash/Fast-style-transfer/blob/master/21anpd.gif?raw=true)

## How to use

This repository is under development. Models are [located here](https://drive.google.com/drive/folders/0B9jhaT37ydSyRk9UX0wwX3BpMzQ?usp=sharing).
Note that this is a bare bones version of [lengstrom's](https://github.com/lengstrom/fast-style-transfer) Implementation.

This achieves around 20 FPS for stylization of 540x600 resolution images. Also check that the transform network has been tinkered with to give maximum throughput but model has not been trained for that network. Use the original network for stylized results. 

This uses PILs ImageGrab feature to simply grab a screenshot and feed it to the network. 

## StyleGUI

The stylegui.py file simply runs the same code but gives you the option of setting the amount of time the code runs for, and the area of your screen it should stylize. It is in development to work with multiple streams like OpenCV (Webcam) or external camera as well. 
