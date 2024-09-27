import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
import logging
import pickle

# Custom local imports
import utils_pytorch as utils
from PET_pytorch2 import PET

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the PET model on different datasets.")
    parser.add_argument("--dataset", type=str, default="jetclass", help="Dataset to use")
    parser.add_argument("--folder", type=str, default="/pscratch/sd/v/vmikuni/PET/", help="Folder containing input files")
    parser.add_argument("--mode", type=str, default="all", help="Loss type to train the model")
    parser.add_argument("--batch", type=int, default=250, help="Batch size")
    parser.add_argument("--epoch", type=int, default=200, help="Max epoch")
    parser.add_argument("--warm_epoch", type=int, default=3, help="Warm up epochs")
    parser.add_argument("--stop_epoch", type=int, default=30, help="Epochs before reducing lr")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--b1", type=float, default=0.95, help="beta1 for Lion optimizer")
    parser.add_argument("--b2", type=float, default=0.99, help="beta2 for Lion optimizer")
    parser.add_argument("--lr_factor", type=float, default=10., help="factor for slower learning rate")
    parser.add_argument("--nid", type=int, default=0, help="Training ID for multiple trainings")
    parser.add_argument("--fine_tune", action='store_true', default=False, help='Fine tune a model')
    parser.add_argument("--local", action='store_true', default=False, help='Use local embedding')
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--drop_probability", type=float, default=0.0, help="Drop probability")
    parser.add_argument("--simple", action='store_true', default=False, help='Use simplified head model')
    parser.add_argument("--talking_head", action='store_true', default=False, help='Use talking head attention')
    parser.add_argument("--layer_scale", action='store_true', default=False, help='Use layer scale in the residual connections')
    return parser.parse_args()

def get_data_loader(flags):
    if flags.dataset == 'top':
        train = utils.TopDataLoader(os.path.join(flags.folder,'TOP', 'train_ttbar.h5'), flags.batch, dist.get_rank(), dist.get_world_size())
        val = utils.TopDataLoader(os.path.join(flags.folder,'TOP', 'val_ttbar.h5'), flags.batch, dist.get_rank(), dist.get_world_size())
    elif flags.dataset == 'opt':
        train = utils.TopDataLoader(os.path.join(flags.folder,'Opt', 'train_ttbar.h5'), flags.batch, dist.get_rank(), dist.get_world_size())
        val = utils.TopDataLoader(os.path.join(flags.folder,'Opt', 'val_ttbar.h5'), flags.batch, dist.get_rank(), dist.get_world_size())
    elif flags.dataset == 'toy':
        train = utils.ToyDataLoader(100000//dist.get_world_size(), flags.batch, dist.get_rank(), dist.get_world_size())
        val = utils.ToyDataLoader(100000//dist.get_world_size(), flags.batch, dist.get_rank(), dist.get_world_size())
    elif flags.dataset == 'tau':
        train = utils.TauDataLoader(os.path.join(flags.folder,'TAU', 'train_tau.h5'), flags.batch, dist.get_rank(), dist.get_world_size())
        val = utils.TauDataLoader(os.path.join(flags.folder,'TAU', 'val_tau.h5'), flags.batch, dist.get_rank(), dist.get_world_size())
    elif flags.dataset == 'qg':
        train = utils.QGDataLoader(os.path.join(flags.folder,'QG', 'train_qg.h5'), flags.batch, dist.get_rank(), dist.get_world_size())
        val = utils.QGDataLoader(os.path.join(flags.folder,'QG', 'val_qg.h5'), flags.batch, dist.get_rank(), dist.get_world_size())
    elif flags.dataset == 'cms':
        train = utils.CMSQGDataLoader(os.path.join(flags.folder,'CMSQG', 'train_qgcms_pid.h5'), flags.batch, dist.get_rank(), dist.get_world_size())
        val = utils.CMSQGDataLoader(os.path.join(flags.folder,'CMSQG', 'val_qgcms_pid.h5'), flags.batch, dist.get_rank(), dist.get_world_size())
    elif flags.dataset == 'h1':
        train = utils.H1DataLoader(os.path.join(flags.folder,'H1', 'train.h5'), flags.batch, dist.get_rank(), dist.get_world_size())
        val = utils.H1DataLoader(os.path.join(flags.folder,'H1', 'val.h5'), flags.batch, dist.get_rank(), dist.get_world_size())
    elif flags.dataset == 'jetclass':
        train = utils.JetClassDataLoader(os.path.join(flags.folder,'JetClass','train'),
                                         flags.batch, dist.get_rank(), dist.get_world_size())
        val = utils.JetClassDataLoader(os.path.join(flags.folder,'JetClass','val'),
                                        flags.batch, dist.get_rank(), dist.get_world_size())

    return train, val

def configure_optimizers(flags, train_loader, lr_factor=1.0):
    scale_lr = flags.lr * np.sqrt(dist.get_world_size())
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=flags.epoch * train_loader.nevts // flags.batch // dist.get_world_size(),
        eta_min=scale_lr / lr_factor / flags.lr_factor
    )
    optimizer = torch.optim.Lion(
        model.parameters(),
        lr=scale_lr / lr_factor,
        weight_decay=flags.wd * lr_factor,
        betas=(flags.b1, flags.b2)
    )
    return optimizer, lr_schedule

def main():
    utils.setup_gpus()
    flags = parse_arguments()

    train_loader, val_loader = get_data_loader(flags)
    
    model = PET(num_feat=train_loader.num_feat,
                num_jet=train_loader.num_jet,
                num_classes=train_loader.num_classes,
                local=flags.local,
                num_layers=flags.num_layers,
                drop_probability=flags.drop_probability,
                simple=flags.simple, layer_scale=flags.layer_scale,
                talking_head=flags.talking_head,
                mode=flags.mode)

    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])

    if flags.fine_tune:
        if dist.get_rank() == 0:
            model_name = utils.get_model_name(flags, flags.fine_tune).replace(flags.dataset,'jetclass').replace('fine_tune','baseline').replace(flags.mode,'all')
            model_path = os.path.join(flags.folder, 'checkpoints', model_name)
            logger.info(f"Loading model weights from {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)

    optimizer_head, scheduler_head = configure_optimizers(flags, train_loader)
    optimizer_body, scheduler_body = configure_optimizers(flags, train_loader, lr_factor=flags.lr_factor if flags.fine_tune else 1.0)

    early_stopping = EarlyStopping(patience=flags.stop_epoch)
    
    for epoch in range(flags.epoch):
        train_loss = train_epoch(model, train_loader, optimizer_body, optimizer_head, flags)
        val_loss = validate_epoch(model, val_loader, flags)
        
        scheduler_head.step()
        scheduler_body.step()
        
        if dist.get_rank() == 0:
            logger.info(f"Epoch {epoch+1}/{flags.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            checkpoint_name = utils.get_model_name(flags, flags.fine_tune,
                                                   add_string=f"_{flags.nid}" if flags.nid > 0 else '')
            checkpoint_path = os.path.join(flags.folder, 'checkpoints', checkpoint_name)
            torch.save(model.state_dict(), checkpoint_path)
            
            if early_stopping(val_loss):
                logger.info("Early stopping")
                break
    
    if dist.get_rank() == 0:
        with open(os.path.join(flags.folder, 'histories', utils.get_model_name(flags, flags.fine_tune).replace(".pth", ".pkl")), "wb") as f:
            pickle.dump({"train_loss": train_loss, "val_loss": val_loss}, f)

def train_epoch(model, dataloader, optimizer_body, optimizer_head, flags):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer_body.zero_grad()
        optimizer_head.zero_grad()
        
        x, y = batch
        x = {k: v.to(device) for k, v in x.items()}
        y = y.to(device)
        
        loss = model.train_step((x, y), optimizer_body)
        
        loss.backward()
        optimizer_body.step()
        optimizer_head.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, flags):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)
            
            loss = model(x, y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()