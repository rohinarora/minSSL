# minSSL - Minimal re-implementation of semi supervised learning algorithms

> Under Construction - April 8, 2021

> add link to ppt here. add some images here to show simclr and moco


## Installation

* Create pytorch env as mentioned in https://rc-docs.northeastern.edu/en/latest/using-discovery/workingwithgpu.html#using-gpus-with-pytorch
```
$ conda activate <env_name>
pip install torchsummary
pip install torchinfo
pip install modelsummary
pip install pytorch-model-summary
```

## Usage

1. Pretrain 
```
python pre_train.py --help #run options
python pre_train.py #default config 
python pre_train.py -data ./datasets --dataset-name stl10 --log-every-n-steps 100 --epochs 100 
```

2. Train/Evaluate 
```
python linear_eval.py --help #run options
python linear_eval.py
```



<!-- 
## Feature Evaluation

Feature evaluation is done using a linear model protocol. 

First, we learned features using SimCLR on the ```STL10 unsupervised``` set. Then, we train a linear classifier on top of the frozen features from SimCLR. The linear model is trained on features extracted from the ```STL10 train``` set and evaluated on the ```STL10 test``` set. 

Check the [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/sthalles/SimCLR/blob/simclr-refactor/feature_eval/mini_batch_logistic_regression_evaluator.ipynb) notebook for reproducibility.

Note that SimCLR benefits from **longer training**.

| Linear Classification      | Dataset | Feature Extractor | Architecture                                                                    | Feature dimensionality | Projection Head dimensionality | Epochs | Top1 % |
|----------------------------|---------|-------------------|---------------------------------------------------------------------------------|------------------------|--------------------------------|--------|--------|
| Logistic Regression (Adam) | STL10   | SimCLR            | [ResNet-18](https://drive.google.com/open?id=14_nH2FkyKbt61cieQDiSbBVNP8-gtwgF) | 512                    | 128                            | 100    | 74.45  |
| Logistic Regression (Adam) | CIFAR10 | SimCLR            | [ResNet-18](https://drive.google.com/open?id=1lc2aoVtrAetGn0PnTkOyFzPCIucOJq7C) | 512                    | 128                            | 100    | 69.82  |
| Logistic Regression (Adam) | STL10   | SimCLR            | [ResNet-50](https://drive.google.com/open?id=1ByTKAUsdm_X7tLcii6oAEl5qFRqRMZSu) | 2048                   | 128                            | 50     | 70.075 | -->


## TODOs
- [x] Pre_training and linear_eval workflow with simCLR
- [ ] KNN test on pre_trained model
- [ ] Reproduce paper claims on small datasets
    - [ ] placeholder1
    - [ ] placeholder1
- [ ] Reproduce paper claims on imageNet size datasets (Once implemented, keep this in background and move on to next. will take days to complete)
    - [ ] placeholder1
    - [ ] placeholder1
- [ ] fp16 benchmarking/benefits
- [ ] Integrate moco codebase
- [ ] Reproduce moco results on small and big datasets

## References
1. https://github.com/sthalles/SimCLR