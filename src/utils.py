import os
import shutil
import torch
import yaml
import argparse
import torchvision
from torch.nn import Linear
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms as transforms

# model_dict={"resnet18": models.resnet18(pretrained=pretrained, num_classes=out_dim),
#             "resnet50": models.resnet50(pretrained=pretrained, num_classes=out_dim)}

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    filename=os.path.join(model_checkpoints_folder, 'config.yml')
    append_write = 'a' if os.path.exists(filename) else 'w'
    with open(filename, append_write) as outfile:
        if append_write=='a':
            outfile.write("\n\nInput config file : \n")
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

def restricted_float(x):
    x=float(x)
    if x < 0.1 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.1, 1.0]"%(x,))
    return x

def update_parser_args(task):
    if task=="pre_train":
        model_names = sorted(name for name in models.__dict__  if name.islower() and not name.startswith("__") and callable(models.__dict__[name])) #to check : local/relative imports working inside models
        parser = argparse.ArgumentParser(description='PyTorch SimCLR Pre Training')
        parser.add_argument('--data', metavar='DIR', default='./datasets', help='path to dataset') #metavar is used in help messages. else default is --foo FOO
        parser.add_argument('--dataset-name', default='stl10', help='dataset name', choices=['stl10', 'cifar10'])
        parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names, help='model architecture: ' + ' | '.join(model_names) +' (default: resnet18)') #metavar replaces all "choices" that would have printed explicitly generally; like in '--dataset-name'
        parser.add_argument('--out_dim', default=128, type=int, help='feature dimension (default: 128)')
        parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
        parser.add_argument('--n-views', default=2, type=int, metavar='N', help='Number of views for contrastive learning training.') #need to test the code with views!=2
        parser.add_argument('--knn_test', action='store_true', help='Evaluate the self trained network via KNN. Boolean. Default False') #The store_true option automatically creates a default value of False. https://stackoverflow.com/questions/8203622/argparse-store-false-if-unspecified
    elif task=="downstream":
        parser = argparse.ArgumentParser(description='PyTorch Self Supervised Learning Linear Evaluation')
        parser.add_argument('-tm','--training_method', required=True, choices=['SSL', 'ImageNet','Scratch'], help='Backend to use for pretraining. Default=None.')
        parser.add_argument('-rd','--run_dir', type=str, default=None,help='Run DIR of pre train network. run DIR has the config file and saved models. Must pass if pre_train_ssl is true') 
        parser.add_argument('-cf','--config_file', type=str, default=None,help='Config file for Supervised Pretrained Network. Must pass if pre_train_ssl is false') 
        parser.add_argument('-downstream','--downstream_task', type=str, required=True,choices=['fine_tune', 'linear_eval'],help='Downstream Task to Run',dest="downstream")
        parser.add_argument('-frac','--fraction_labels', type=restricted_float,help='Fraction of train labels to be used',dest="frac",default=None)
    parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float, metavar='LR', help='initial learning rate (default=0.0003)', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=0.0008, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--log-every-n-steps', default=100, type=int, help='Log every n steps')
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training (default=42)')
    parser.add_argument('--gpu-index', default=0, type=int, help='GPU index.')
    parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256). \nIn multi GPU setting this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-e','--epochs', default=200, type=int, metavar='E', help='number of total epochs to run (default: 200)')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 32)')
    parser.add_argument('-cmt','--comment', default="",  type=str, help='Exp comment. Appended to run_dir name',dest='comment')
    parser.add_argument('-opt','--optimizer', type=str, default="Adam",choices=['Adam', 'AdamW'],help='Optimizer to use',dest="opt")
    args = parser.parse_args()
    if task=="downstream": 
        if args.training_method=='SSL' and (args.run_dir is None):
            parser.error("This mode requires --run_dir flag. See help for more.")
        elif (args.training_method=='ImageNet' or args.training_method=='Scratch') and (args.config_file is None):
            parser.error("Supervised Pretrained/Train from scratch requires --config_file flag. See help for more.")
    update_cuda_flags(args)
    return args

def get_config(args):
    if args.training_method =="Scratch" or args.training_method =="ImageNet":
        config_file=args.config_file
        with open(os.path.join(config_file)) as file:
            config = yaml.load(file)
    elif args.training_method =="SSL": # use SSL pre_trained network
        config_file=args.run_dir+"/config.yml"
        with open(os.path.join(config_file)) as file:
            config = yaml.load(file)
    return config


def get_dataloader(args, train_ds,test_ds):
    return torch.utils.data.DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True), \
           torch.utils.data.DataLoader(dataset=test_ds, batch_size=2*args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
