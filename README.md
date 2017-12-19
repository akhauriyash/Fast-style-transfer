# Fast Style Transfer

Implementation of [Instance Normalization](https://arxiv.org/abs/1607.08022) and a generator network for one shot stylization. 

## How to use

This repository is under development. Models are [located here](https://drive.google.com/drive/folders/0B9jhaT37ydSyRk9UX0wwX3BpMzQ?usp=sharing).
Note that this is a bare bones version of [lengstrom's](https://github.com/lengstrom/fast-style-transfer) Implementation.

This achieves around 20 FPS for stylization of 540x600 resolution images. Also check that the transform network has been tinkered with to give maximum throughput but model has not been trained for that network. Use the original network for stylized results. 

This uses PILs ImageGrab feature to simply grab a screenshot and feed it to the network. 
