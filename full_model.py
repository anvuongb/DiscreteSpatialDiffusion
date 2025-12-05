import os
import argparse
import multiprocessing
from functools import partial

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from score_models import ScoreModel, NCSNpp
from ema_pytorch import EMA

from faster_diffusion_dataset import full_diffusion_dataset
import glob


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    parser.add_argument('--r', type=int, default=160, help='Hopping rate in each direction')
    parser.add_argument('--t_size', type=int, default=2_000, help='Time size')
    parser.add_argument('--schedule', type=str, default='blackout', help='Noise scheduler class')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--loss', type=str, choices=['L1', 'L2', 'L1_norm', 'blackout'], default='blackout', help='Loss function')
    parser.add_argument('--accum_grads', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of samples in a batch')
    parser.add_argument('--num_workers', type=int, default=100, help='Number of workers for data loading')
    parser.add_argument('--gpu_num', type=int, default=None , help='GPU number to use')
    parser.add_argument('--num_devices', type=int, default=1, help='Number of devices (GPUs) to use') 
    parser.add_argument('--data', type=str, choices=['mnist', 'cifar10', 'cifar_gray', 'celebA'], default='cifar10', help='Dataset to use')
    parser.add_argument('--notes', type=str, default='Comments go here', help='Ans')
    parser.add_argument('--resume_version', type=str, default=None, help='Version to resume from (will use latest)')
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Explicit ckpt path to resume from')
    parser.add_argument('--PBC', type=int, default=1, help='Periodic Boundary Condition (1/0)')
    parser.add_argument('--circular_padding', type=int, default=1, help='use circular padding in the NN (1) or not (0)')
    
    return parser.parse_args()

def set_circular_padding(model):
    """
    Recursively set padding_mode to 'circular' for all Conv2d layers in the model.
    """
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Conv2d):
            module.padding_mode = 'circular'
        elif isinstance(module, torch.nn.Sequential) or isinstance(module, torch.nn.Module):
            set_circular_padding(module) 


class DiffusionModel(pl.LightningModule):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        print(self.hparams)
        
    
        self.lr = self.hparams.lr
        in_ch = self.hparams.in_ch
        
        # Yang Song's NCSN++ architecture
        config = {
            'channels': in_ch,
            'dimensions':2,
            'progressive':'none',
            'progressive_input':'residual',
            'combine_method':'sum',
            'init_scale':1e-5, # original ncspp used 1e-10 
            'num_res_blocks':4,
            'dropout':0.1
        }
        net = NCSNpp(**config)
        #net = NCSNpp(channels=in_ch, dimensions=2, nf=128, ch_mult=[1, 2, 2, 2], dropout=0.1)
        
        # Adjust the model
        for name, module in list(net.named_modules()):
            if isinstance(module, torch.nn.Conv2d) and module.out_channels == in_ch:
                new_conv = torch.nn.Conv2d(module.in_channels, 4 * in_ch,
                                           module.kernel_size, module.stride,
                                           module.padding, module.dilation, module.groups,
                                           module.bias is not None, module.padding_mode)
        
                new_module = torch.nn.Sequential(new_conv, torch.nn.Softplus())
        
                parent_name, child_name = name.split('.')[:2]
                getattr(net, parent_name)[int(child_name)] = new_module
                getattr(net, parent_name)[int(child_name)].weight = new_module[0].weight
        
        if self.hparams.circular_padding==1:
            set_circular_padding(net)

        self.model = ScoreModel(model=net, beta_min=0.1, beta_max=20)
        
        self.ema = EMA(
                        self.model,
                        beta=0.9999,            # Exponential moving average factor
                        update_after_step=100,  # Start updating after 100 steps
                        update_every=10,        # Update EMA every 10 steps
                    )

    
    def on_before_zero_grad(self, optimizer):
        self.ema.update()

   
    def forward(self, ts, out_im):
        return self.model.model(ts, out_im)
 
    
    def training_step(self, batch, batch_idx):
        
        out_im, reverse_rates, ts, dts = batch
        yhat = self(ts, out_im)
        
        if self.hparams.loss == 'L1':
            l1_loss = torch.mean(torch.abs(yhat - reverse_rates))
            loss = l1_loss
            
        elif self.hparams.loss == 'L2':
            loss = torch.mean((yhat - reverse_rates) ** 2)
            l1_loss = torch.mean(torch.abs(yhat - reverse_rates))
            
            
        elif self.hparams.loss == 'blackout': 
            
            out = out_im.repeat_interleave(4, dim=1)
            pred = yhat*out
            loss = torch.mean(dts[:,:,None,None]*(pred - reverse_rates*torch.log(torch.clamp(pred, min=1e-8))))
            l1_loss = torch.mean(torch.abs(pred - reverse_rates))
        
        else: 
            raise NotImplementedError
        
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('L1_loss', l1_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
   
    
def main():
        
    args = parse_args()
    args_dict = vars(args)
    
    dataset = full_diffusion_dataset(data=args.data,
                                     r=args.r, 
                                     t_size=args.t_size, 
                                     schedule=args.schedule,
                                     PBC=(args.PBC==1))
    
    args_dict['in_ch'] = dataset.in_ch
    
    dataloader =  DataLoader( dataset, 
                              batch_size = args.batch_size, 
                              pin_memory = torch.cuda.is_available(), 
                              shuffle = True,
                              num_workers = args.num_workers,
                              persistent_workers = args.num_workers > 0,
                         )
    
    diffusion_model = DiffusionModel(**args_dict)
    
    
    cbs = [ModelCheckpoint(filename = "train-{step:07d}", 
                           every_n_train_steps = 10_000, 
                           save_top_k = -1,
                           save_on_train_epoch_end = False),
           
           ModelCheckpoint(filename = "best-{step:07d}", 
                           monitor = "train_loss",
                           save_top_k = 1,
                           every_n_train_steps = 1_000,
                           save_last = False),
           ]
    


    device = f'cuda:{args.gpu_num}' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    trainer = pl.Trainer(max_steps = 20_000_000, 
                         gradient_clip_val = 1,
                         callbacks = cbs, 
                         accumulate_grad_batches = args.accum_grads,  
                         log_every_n_steps = args.accum_grads,
                         accelerator = 'auto',
                         devices = args.num_devices if args.num_devices > 1 else [args.gpu_num] if args.gpu_num is not None else 1
                         )
    

    if args.resume_ckpt is not None:
        ckpt_path = args.resume_ckpt
    elif args.resume_version is not None:
        checkpoint_dir = f'lightning_logs/version_{args.resume_version}/checkpoints/'
        checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
        checkpoint_paths = sorted(checkpoint_paths, key=lambda x: int(x.split('step=')[-1].split('.ckpt')[0]), reverse=True)
        ckpt_path = checkpoint_paths[0] if checkpoint_paths else None
    else:
        ckpt_path = None

    print(f'Resuming from {ckpt_path}')

    trainer.fit(diffusion_model, dataloader, ckpt_path=ckpt_path)    
    
    
if __name__ == "__main__":
    #multiprocessing.set_start_method('spawn', force=True)
    main()