import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import PreTrain_NN
from simclr import SimCLR
import random
import os
import numpy as np
from utils import update_cuda_flags, update_parser_args

torch.manual_seed(42) #seed torch
random.seed(42) #seed python
np.random.seed(42) #seed numpy. try to only use randomness from pytorch. try not to mix numpy and pytorch for RNG. some applications and libraries further may use NumPy Random Generator objects, not the global RNG, and those will need to be seeded consistently as well.. save the RNG state when and if checkpointing. More, later : https://pytorch.org/docs/stable/notes/randomness.html https://pytorch.org/docs/stable/notes/faq.html

def main():
    args=update_parser_args()    
    CL_dataset = ContrastiveLearningDataset(args.data)
    train_loader = torch.utils.data.DataLoader(dataset=CL_dataset.get_dataset(args.dataset_name, args.n_views), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    simclr_model = PreTrain_NN(base_model=args.arch, out_dim=args.out_dim)
    optimizer = torch.optim.Adam(simclr_model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    with torch.cuda.device(args.gpu_index): #  It’s a no-op if the 'gpu_index' argument is a negative integer or None. May need to update this for multiGPU?
        simclr_pt = SimCLR(model=simclr_model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr_pt.train(train_loader)

if __name__ == "__main__":
    main()