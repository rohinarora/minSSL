import torch
from utils import get_downstream_model, get_dataloader, get_config, update_parser_args_downstream_eval
from torch.utils.tensorboard import SummaryWriter
from downstream import Downstream_Eval
from linear_eval_dataset import CustomDataset

def main():
    args=update_parser_args_downstream_eval()
    config=get_config(args)
    downstream_dataset = CustomDataset(config.data)
    train_ds,test_ds=downstream_dataset.get_dataset(config.dataset_name)
    train_loader,test_loader = get_dataloader(args, train_ds,test_ds)
    model=get_downstream_model(args,config)
    optimizer = eval("torch.optim."+args.opt)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    with torch.cuda.device(args.gpu_index):
        downstream_task = Downstream_Eval(kind=args.downstream,model=model, optimizer=optimizer, scheduler=scheduler, args=args, config=config)
        downstream_task.train(train_loader,test_loader)

if __name__ == "__main__":
    main()