import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
import random
import numpy as np



torch.manual_seed(42) #seed
random.seed(42) #seed python
np.random.seed(42) #seed numpy. try to only use randomness from pytorch. try not to mix numpy and pytorch for RNG
# some applications and libraries further may use NumPy Random Generator objects, not the global RNG, and those will need to be seeded consistently as well.
#save the RNG state when and if checkpointing

# More, later : https://pytorch.org/docs/stable/notes/randomness.html https://pytorch.org/docs/stable/notes/faq.html

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

#what is the pythonic stuff going on in above line, how is the local/relative imports working inside models

parser = argparse.ArgumentParser(description='PyTorch SimCLR Pre Training')
parser.add_argument('--data', metavar='DIR', default='./datasets', #metavar is used in help messages. else default is --foo FOO
                    help='path to dataset')
parser.add_argument('--dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', #metavar replaces all "choices" that would have printed explicitly generally; like in '--dataset-name'
                    choices=model_names, help='model architecture: ' + ' | '.join(model_names) +' (default: resnet18)')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-e','--epochs', default=200, type=int, metavar='E',
                    help='number of total epochs to run (default: 200)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256). \nIn multi GPU setting \
                        this is the total batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate (default=0.0003)', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training (default=42)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='GPU index.')
parser.add_argument('--knn_test', action='store_true', help='Evaluate the self trained network via KNN. Boolean. Default False') #The store_true option automatically creates a default value of False. https://stackoverflow.com/questions/8203622/argparse-store-false-if-unspecified



def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True #https://discuss.pytorch.org/t/what-is-the-differenc-between-cudnn-deterministic-and-cudnn-benchmark/38054
        cudnn.benchmark = False #For reproducibility. "good practice to turn off cudnn.benchmark when turning on cudnn.deterministic"
        torch.set_deterministic(True) #set_deterministic is equivalent to use_deterministic_algorithms(bool)
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(args.data)

    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    
    with torch.cuda.device(args.gpu_index): #  It’s a no-op if the 'gpu_index' argument is a negative integer or None. May need to change this for multiGPU?
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
