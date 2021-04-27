import os
import shutil
import torch
import yaml
import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms as transforms

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_stl10_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.STL10('./data', split='train', download=download, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.STL10('./data', split='test', download=download, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size, num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_cifar10_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.CIFAR10('./data', train=True, download=download, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, drop_last=False, shuffle=shuffle)
    test_dataset = datasets.CIFAR10('./data', train=False, download=download, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size, num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def update_cuda_flags(args):
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = 'cuda' # args.device = torch.device('cuda') #doesnt dump into config.yml well
        cudnn.deterministic = True #https://discuss.pytorch.org/t/what-is-the-differenc-between-cudnn-deterministic-and-cudnn-benchmark/38054
        cudnn.benchmark = False #For reproducibility. "good practice to turn off cudnn.benchmark when turning on cudnn.deterministic"
        torch.set_deterministic(True) #set_deterministic is equivalent to use_deterministic_algorithms(bool)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        args.device = 'cpu'
        args.gpu_index = -1

def update_parser_args_linear_eval():
    parser = argparse.ArgumentParser(description='PyTorch SimCLR Linear Eval')
    parser.add_argument('-pt_ssl','--pre_train_ssl', action='store_true', help='What backend to use for pretraining. Boolean. Default pt_ssl=False, i.e, ImageNet pretrained network. ')
    parser.add_argument('-rd','--run_dir', type=str, default=None,help='Run DIR of pre train network. run DIR has the config file and saved models. Must pass if pre_train_ssl is true') # add some conditions to enforce this check
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    update_cuda_flags(args)
    return args

def update_parser_args():
    model_names = sorted(name for name in models.__dict__  if name.islower() and not name.startswith("__") and callable(models.__dict__[name])) #to check : local/relative imports working inside models
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
    parser.add_argument('--n-views', default=2, type=int, metavar='N', #need to test the code with views!=2
                        help='Number of views for contrastive learning training.')
    parser.add_argument('--gpu-index', default=0, type=int, help='GPU index.')
    parser.add_argument('--knn_test', action='store_true', help='Evaluate the self trained network via KNN. Boolean. Default False') #The store_true option automatically creates a default value of False. https://stackoverflow.com/questions/8203622/argparse-store-false-if-unspecified
    args = parser.parse_args()
    update_cuda_flags(args)
    return args