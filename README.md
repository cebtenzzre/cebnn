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
5. Train models:
```
train 1.0 10 20e-4 --sublayers=.91 --resample='+:8:all' --classifier-dropout=.65 --criterion=digamma
train 2.0 10 20e-4 --sublayers=.91 --resample='+:8:all' --classifier-dropout=.65 --criterion=digamma --seed=44
train 3.0 15 8e-4 --cv-fold=0 --sublayers=.4 --resample=none --inner-dropout=.05 --classifier-dropout=.65 --criterion=digamma --optimizer=aadamw01 --gamma=.1 --sch-period=5
```
6. Evaluate the models:
```
mcc nets/trial1/*.torch
```
7. Generate the binary accuracy information:
```
get_correct fbeta nets/trial1/*.torch
```
8. Run majority voting trials:
```
time ./best_majvote.py 2 0 1 nets/trial1/correctfbeta/*
```
9. Generate test binary accuracy information:
```
get_correct_test fbeta nets/trial1/*.torch
```
10. Evaluate the selected majority vote for each label:
```
./best_majvote_eval.py 0 0 nets/trial1/correctfbeta_test/rexnet_200_{1,2,3}.0.torch_correct.pkl
./best_majvote_eval.py 0 1 nets/trial1/correctfbeta_test/rexnet_200_{1,2,3}.0.torch_correct.pkl
```
11. Run inference on a given dataset for each model:
```
./infer.py nets/trial1/rexnet_200_1.0.torch infer/trial1/file_list.txt infer/trial1/file_list_results_trial1_rexnet_200_1.0_1.pkl; done
./infer.py nets/trial1/rexnet_200_2.0.torch infer/trial1/file_list.txt infer/trial1/file_list_results_trial1_rexnet_200_2.0_1.pkl; done
./infer.py nets/trial1/rexnet_200_3.0.torch infer/trial1/file_list.txt infer/trial1/file_list_results_trial1_rexnet_200_3.0_1.pkl; done
```
12. (A) Compute majority vote and print results:
```
./infer_print_majvote.py "$(<data/trial1/classes.txt)" 0 0 $(for n in rexnet_200_{1,2,3}.0; do echo infer/trial1/file_list_results_trial1_${n}_1.pkl nets/trial1/correctfbeta/${n}.torch_correct.pkl; done) >infer/trial1/mydata_results_trial1_majvote_1.csv
```
12. (B) Or, print the results for a single net/label with a given threshold:
```
./infer_print.py "$(<data/trial1/classes.txt)" 1 .4837 infer/trial1/file_list_results_trial1_rexnet_200_1.0_1.pkl >infer/trial1/file_list_results_trial1_rexnet_200_1.0_1.csv
```
