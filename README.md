# Real-time Arbitrary Style Transfer with Adaptive Instance Normalization

This repository contains the implementation of a real-time arbitrary style transfer using Adaptive Instance Normalization (AdaIN) from the paper Xun Huang, & Serge Belongie, (2017). *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*. This project was developed as part of the Generative Modelling for Images class at MVA (Mathématiques, Vision, Apprentissage) Master at École Normale Supérieure Paris-Saclay. The course was taught by Arthur Leclaire and Bruno Galerne.

## Team Members
- Hippolyte Pilchen ([forename.lastname@polytechnique.edu](mailto:forename.lastname@polytechnique.edu))
- Lucas Gascon

## Introduction
Arbitrary style transfer aims to apply the artistic style of one image (the style image) to another image (the content image) while preserving the content of the latter. This project explores real-time style transfer by training a decoder.

## Methodology
The core technique used in this project is Adaptive Instance Normalization (AdaIN). AdaIN aligns the mean and variance of the content features with those of the style features, effectively transferring the style to the content image. We implemented this technique using PyTorch and developped a new architectures using skip-connections between the encoder and the decoder. 

## Requirements
- torch
- torchvision
- opencv-python
- Pillow>=8.3.2
- matplotlib
- tqdm
- wandb



## Usage
1. First, load wikiart and MS-COCO datasets in 'data/Style_train' and 'data/Content_train'.
2. To train the decoder with concatenation of the content as skip-connections run:
   ```
   python train.py --normed_vgg --cat_skip --skip_type content  --alpha 1.0 --style_weight 10 --train_content_imgs data/Content_train --train_style_imgs data/Style_train
   ```
3. To test the pretrained model and compare it to Gatys *et al.* algorithm run the cells of the notebook

## Results

Results for various style on various image contents using our implementation of skip-connected AdaIN architecture by concatenating content feature maps. The model has been trained for $30$ epochs. $\alpha$ has been set to $1$ and the style weight to $10$.

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)




