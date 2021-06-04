# resnet_cifar100
## Requirements

This is my experiment eviroument
- python3.6
- pytorch1.6.0+cu101
- tensorboard 2.2.2(optional)

## How to run?

### 1. enter directory
```bash
$ cd pytorch-cifar100-all
```
### 2. train the model

```bash
$ python3 train.py -net [resnet50/resnet50_wn/resnet50_wide] -gpu
```
### 3. test the model
Test the model using test.py
```bash
$ python3 test.py -net [resnet50/resnet50_wn/resnet50_wide] -weights path_to_model_weights_file
```
