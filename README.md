# cebnn

A personal project I use to train neural networks. See cmds.zsh for the kinds of things it can do.

## Setup

1. Install python 3.10
2. Create a venv:
```
python3.10 -m venv .venv --system-site-packages
```
3. Activate the venv:
```
source .venv/bin/activate
```
4. Install the requirements:
```
pip install -r requirements.txt
```
torchvision may need to be installed outside of pip if it fails to build.

## Basic usage example

1. Activate the environment (assumes zsh):
```
source cmds.zsh
```
2. Gather images to train on:
```
time ./gather_images.py data/trial1 ::/path/to/negative_class :labela:/path/to/a_class :labelb:/path/to/b_class
```
3. Setup data:
```
./setup_data.sh data/trial1 /mnt/big_hdd/scaled
```
4. Edit cmds.zsh to point to the dataset (`dataset=trial1`)
5. Train a model:
```
( model=rexnet_200; train 1.0 10 20e-4 --sublayers=.91 --resample='+:8:all' --classifier-dropout=.65 --criterion=digamma; )
```
6. Evaluate the model:
```
mcc nets/trial1/rexnet_200_1.0.torch
```
