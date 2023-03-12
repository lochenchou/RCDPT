import os
import torch
import torch.nn as nn
import numpy as np
import utils
import random
from tqdm import tqdm

from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from utils import PolynomialLRDecay
from nuscenes.nuscenes import NuScenes
from dataloader import nusc_loader_mer
from engine import train_one_epoch, validation
from model import networks


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    help="which model to use",
                    choices=['DPT', 'RCDPT', 'DPT_early', 'DPT_late'],
                    default='RCDPT')
parser.add_argument('--nusc_datapath', type=str,
                    default='/datasets/nuscenes/v1.0-trainval')
parser.add_argument('--nusc_version', type=str,
                    default='v1.0-trainval')

args = parser.parse_args()
print(args)

print('Start training with model={}'.format(args.model))

BATCH_SIZE = 4
EPOCHS = 60
LR = 0.0001
END_LR = 0.00001
POLY_POWER = 0.9
LR_PATIENCE = 10
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
MAX_ITER = 300000
WORKERS = 4
SEED = 1984
PRINT_FREQ = 250

# set random seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


output_dir = os.path.join('./result','{}'.format(args.model))
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'valid')
logdir = os.path.join(output_dir, 'log')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
print('OUTPUT_DIR = {}'.format(output_dir))


if args.model == 'DPT':
    model = networks.DPT()
elif args.model == 'RCDPT':
    model = networks.RCDPT()
elif args.model == 'DPT_early':
    model = networks.DPT_early()
elif args.model == 'DPT_late':
    model = networks.DPT_late()
    
if args.model == 'DPT':
    modality = 'single'
else:
    modality = 'multi'

# load MER data loader

print('Load nusc data, data path = {}, version = {}'.format(args.nusc_datapath, args.nusc_version))
nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_datapath, verbose=True)
train_loader = nusc_loader_mer.init_data_loader(mode='train', shuffle=True, batch_size=BATCH_SIZE, num_workers=WORKERS, nusc=nusc)
val_loader = nusc_loader_mer.init_data_loader(mode='val', shuffle=False, batch_size=BATCH_SIZE, num_workers=WORKERS, nusc=nusc)


print('GPU number: {}'.format(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if GPU number > 1, then use multiple GPUs
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)


# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = PolynomialLRDecay(optimizer, max_decay_steps=MAX_ITER, end_learning_rate=END_LR, power=POLY_POWER)
        
logger = SummaryWriter(logdir)
epochbar = tqdm(total=EPOCHS)


for epoch in range(EPOCHS):
    train_one_epoch(device, train_loader, model, train_dir, optimizer, epoch, logger, modality, PRINT_FREQ)
    validation(device, val_loader, model, val_dir, epoch, logger, modality)
    epochbar.update(1)

    # save model and checkpoint every epoch
    ckpt_filename = os.path.join(output_dir, 'ckpt_{}.tar'.format(str(epoch)))
    torch.save({'model_state_dict': model.state_dict(),}, ckpt_filename)




