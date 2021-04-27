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
from utils import save_config_file, accuracy, save_checkpoint, get_stl10_data_loaders , get_cifar10_data_loaders, update_parser_args_linear_eval
from torch.utils.tensorboard import SummaryWriter
import glob


def main():
    args=update_parser_args_linear_eval()
    if not args.pre_train_ssl: #use ImageNet pretrained network
        pass
    else: # use SSL pre_trained network
        if not args.run_dir:
            print ("must pass run_dir with pre_train_ssl flag True") #raise some exception
            return
        config_file=args.run_dir+"/config.yml"
        with open(os.path.join(config_file)) as file:
            config = yaml.load(file)
            if config.arch == 'resnet18': #make the below lines work with any "torchvision" model, any number of classes, any dataset. improve this
                model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(args.device)
            elif config.arch == 'resnet50':
                model = torchvision.models.resnet50(pretrained=False, num_classes=10).to(args.device)
            checkpoint_file=[name for name in glob.glob(args.run_dir+'/checkpoint*')][-1] #picks the last checkpoint if there are > 1
            checkpoint = torch.load(checkpoint_file, map_location=args.device)
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('backbone.'):
                    if k.startswith('backbone') and not k.startswith('backbone.fc'): state_dict[k[len("backbone."):]] = state_dict[k] # remove prefix
                del state_dict[k]
            log = model.load_state_dict(state_dict, strict=False)
            assert log.missing_keys == ['fc.weight', 'fc.bias']
        if config.dataset_name == 'cifar10':
            train_loader, test_loader = get_cifar10_data_loaders(download=True)
        elif config.dataset_name == 'stl10':
            train_loader, test_loader = get_stl10_data_loaders(download=True)
        print("Dataset:", config.dataset_name) #logger
        for name, param in model.named_parameters(): # freeze all layers but the last fc
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0008)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    epochs = 100
    for epoch in range(epochs):
        top1_train_accuracy = 0 #train score per epoch
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        top1_train_accuracy /= (counter + 1)
        top1_accuracy = 0 #test score per epoch
        top5_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)
            logits = model(x_batch)
            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]
        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")

if __name__ == "__main__":
    main()