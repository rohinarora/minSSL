# PyTorch SimCLR: A Simple Framework for Contrastive Learning of Visual Representations


> add link to ppt here. add some images here to show simclr

![Image of SimCLR Arch](https://sthalles.github.io/assets/contrastive-self-supervised/cover.png)

## Installation

* Create pytorch env as mentioned in https://rc-docs.northeastern.edu/en/latest/using-discovery/workingwithgpu.html#using-gpus-with-pytorch
```
$ conda activate <env_name>

```

## Usage

```
python run.py
```

* Refer ```python run.py --help``` to see the run options. Allows to easily update the hyperparameters

Example : 
```
python run.py -data ./datasets --dataset-name stl10 --log-every-n-steps 100 --epochs 100 
```

> onwards. take inspiration. make something like below


## Feature Evaluation

Feature evaluation is done using a linear model protocol. 

First, we learned features using SimCLR on the ```STL10 unsupervised``` set. Then, we train a linear classifier on top of the frozen features from SimCLR. The linear model is trained on features extracted from the ```STL10 train``` set and evaluated on the ```STL10 test``` set. 

Check the [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/sthalles/SimCLR/blob/simclr-refactor/feature_eval/mini_batch_logistic_regression_evaluator.ipynb) notebook for reproducibility.

Note that SimCLR benefits from **longer training**.

| Linear Classification      | Dataset | Feature Extractor | Architecture                                                                    | Feature dimensionality | Projection Head dimensionality | Epochs | Top1 % |
|----------------------------|---------|-------------------|---------------------------------------------------------------------------------|------------------------|--------------------------------|--------|--------|
| Logistic Regression (Adam) | STL10   | SimCLR            | [ResNet-18](https://drive.google.com/open?id=14_nH2FkyKbt61cieQDiSbBVNP8-gtwgF) | 512                    | 128                            | 100    | 74.45  |
| Logistic Regression (Adam) | CIFAR10 | SimCLR            | [ResNet-18](https://drive.google.com/open?id=1lc2aoVtrAetGn0PnTkOyFzPCIucOJq7C) | 512                    | 128                            | 100    | 69.82  |
| Logistic Regression (Adam) | STL10   | SimCLR            | [ResNet-50](https://drive.google.com/open?id=1ByTKAUsdm_X7tLcii6oAEl5qFRqRMZSu) | 2048                   | 128                            | 50     | 70.075 |
