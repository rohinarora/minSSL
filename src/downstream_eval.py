import torch
from utils import get_dataloader, get_config, update_parser_args
from downstream import Downstream_Eval
from linear_eval_dataset import CustomDataset
from models.downstream_nn import Downstream_NN
import random

torch.manual_seed(42) #seed torch
random.seed(42) #seed python


def main():
    args=update_parser_args(task="downstream")
    config=get_config(args)
    downstream_dataset = CustomDataset(config.data)
    train_ds,test_ds=downstream_dataset.get_dataset(config.dataset_name)
    num_classes=len(train_ds.classes)
    if args.frac: #use fraction of train labels
        samples=list(range(0, len(train_ds)))
        random.shuffle(samples)
        samples=samples[:int(args.frac*len(train_ds))] #might want to rethink when classes are imbalanced
        train_ds=torch.utils.data.Subset(train_ds, samples) #https://stackoverflow.com/a/58703467/5536853 https://discuss.pytorch.org/t/how-to-get-a-part-of-datasets/82161
    train_loader,test_loader = get_dataloader(args, train_ds,test_ds)
    model=Downstream_NN(args,config,num_classes=num_classes)
    optimizer = eval("torch.optim."+args.opt)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    with torch.cuda.device(args.gpu_index):
        downstream_task = Downstream_Eval(kind=args.downstream,model=model, optimizer=optimizer, scheduler=scheduler, args=args, config=config)
        downstream_task.train(train_loader,test_loader)

if __name__ == "__main__":
    main()