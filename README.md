## minSSL - A Minimal Library for Self-Supervised Learning


### Installation

* Create pytorch env as mentioned in https://rc-docs.northeastern.edu/en/latest/using-discovery/workingwithgpu.html#using-gpus-with-pytorch
```
$ conda activate <env_name>

```

### Usage

1. Pretrain 
```
python pre_train.py --help #run options
python pre_train.py #default config 
python pre_train.py -data ./datasets --dataset-name stl10 --log-every-n-steps 100 --epochs 100 
```

2. Evaluate on downstream tasks 
```
python downstream_eval.py --help #run options
```

* Currently supports SimCLR and MoCo. More SSL algorithms coming soon !


## Authors
* Rohin Arora, [Syed Shahbaaz Ahmed](shahbaazsyed1@gmail.com) and [Varun Sahasrabudhe](https://github.com/vsahasrabudhe96)