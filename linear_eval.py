import torch
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import argparse
from utils import get_dataloader, get_config, save_config_file, accuracy, save_checkpoint, get_stl10_data_loaders , get_cifar10_data_loaders, update_parser_args_linear_eval, get_linear_eval_model, get_dataset
from torch.utils.tensorboard import SummaryWriter
import glob
from torch.nn import Linear
from le import Linear_Eval
from linear_eval_dataset import CustomDataset

def main():
    args=update_parser_args_linear_eval()
    config=get_config(args)
    LE_dataset = CustomDataset(config.data)
    train_ds,test_ds=LE_dataset.get_dataset(config.dataset_name)
    train_loader,test_loader = get_dataloader(args, train_ds,test_ds)
    model=get_linear_eval_model(args,config)
    optimizer = eval("torch.optim."+args.opt)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    with torch.cuda.device(args.gpu_index):
        lin_eval = Linear_Eval(model=model, optimizer=optimizer, scheduler=scheduler, args=args, config=config)
        lin_eval.train(train_loader,test_loader)

if __name__ == "__main__":
    main()