import logging
import os
import sys
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
import time

torch.manual_seed(42)




class Linear_Eval():
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter("runs_le")
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def train(self,train_loader,test_loader):
        start=time.time()
        scaler = GradScaler(enabled=self.args.fp16_precision) #todo: add autocast below as needed
        save_config_file(self.writer.log_dir, self.args)
        logging.info(f"Start training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu. args.disable_cuda flag: {self.args.disable_cuda}.")
        for epoch in range(self.args.epochs):
            top1_train_accuracy = 0 #train score per epoch
            for counter, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)
                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)
                top1 = accuracy(logits, y_batch, topk=(1,))
                top1_train_accuracy += top1[0]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            top1_train_accuracy /= (counter + 1)
            top1_accuracy = 0 #test score per epoch
            top5_accuracy = 0
            for counter, (x_batch, y_batch) in enumerate(test_loader):
                x_batch = x_batch.to(self.args.device)
                y_batch = y_batch.to(self.args.device)
                logits = self.model(x_batch)
                top1, top5 = accuracy(logits, y_batch, topk=(1,5))
                top1_accuracy += top1[0]
                top5_accuracy += top5[0]
            top1_accuracy /= (counter + 1)
            top5_accuracy /= (counter + 1)
            print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
        logging.info("Training has finished.")
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        end=time.time()
        logging.info(f"Runtime is  {end-start}.")