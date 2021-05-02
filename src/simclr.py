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
import socket
from datetime import datetime

torch.manual_seed(42)

class SimCLR():
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(log_dir = os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname() + self.args.comment))
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0) #labels.shape = [2*batch_size]. Every image has a unique label. Different augmented versions of same image get same labels 
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() #labels.shape [2*batch_size,2*batch_size].
        labels = labels.to(self.args.device) 
        features = F.normalize(features, dim=1) #normalize each image feature
        similarity_matrix = torch.matmul(features, features.T) #features.shape=[512,128]. similarity_matrix.shape=[n_views*batch_size,n_views*batch_size]. It has similarity of every image with other image. 50% wasteful compute, but its vectorized. # assert similarity_matrix.shape == labels.shape
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)  # discard the main diagonal from both: labels and similarity_matrix. similarity_matrix has 1's in diagonal, since features are normalized
        labels = labels[~mask].view(labels.shape[0], -1) #512*512-512 elements. [512, 511] after reshape
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) #[512, 511]. # assert similarity_matrix.shape == labels.shape
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1) #[512, 1]. # select positives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) #[512, 510]. # select negatives
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):
        start=time.time()
        scaler = GradScaler(enabled=self.args.fp16_precision)
        save_config_file(self.writer.log_dir, self.args)
        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu. args.disable_cuda flag: {self.args.disable_cuda}.")
        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader): #dont need true labels
                images = torch.cat(images, dim=0) #input has two augmmented version of same images in list : [images[0],images[1]]. images[0].shape = torch.Size([256, 3, 96, 96]) ; images[1].shape = torch.Size([256, 3, 96, 96])
                images = images.to(self.args.device) #2*batch_size,3,96,96
                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images) #features.shape = [2*batch_size, out_dim]
                    logits, labels = self.info_nce_loss(features) #logits.shape = (512, 511) ; labels.shape=torch.Size([512])
                    loss = self.criterion(logits, labels) #loss is a single number. https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5)) #top1 accuracy could have been summed over entire epoch rather than recorded every minibatch. too late to change. later
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                n_iter += 1
            if epoch_counter >= 10: # warmup for the first 10 epochs
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
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
