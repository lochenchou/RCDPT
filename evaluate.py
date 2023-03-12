import os
import torch
import torch.nn as nn
import numpy as np
import utils
import random
import time
from tqdm import tqdm

from dataloader import nusc_loader_mer_eval
from metrics import AverageMeter, Result, compute_errors
from model import networks


def write_result(file, average, caption):
    file.write('\n**** {} ***\n'.format(caption))
    file.write('RMSE \tRMSElog iRMSE \tiMAE \tAbsRel \tSqRel \tSILog \tD1 \t\tD2 \t\tD3\n'
               '{avg.rmse:.3f}\t{avg.rmse_log:.3f}\t{avg.irmse:.3f}\t{avg.imae:.3f}'
               '\t{avg.absrel:.3f}\t{avg.squared_rel:.3f}\t{avg.silog:.3f}'
               '\t{avg.delta1:.3f}\t{avg.delta2:.3f}\t{avg.delta3:.3f}\n'.format(avg=average.average()))

def validation(device, data_loader, model, output_dir, file, modality):
    avg80_sparse = AverageMeter()
    avg50_sparse = AverageMeter()

    model.eval()
    
    end = time.time()
    print('START EVALUATION with data LEN={}'.format(len(data_loader)))
    
    skip = int(len(data_loader) / 10)
    
    evalbar = tqdm(total=len(data_loader))
    
    for i, data in enumerate(data_loader):
        _rgb = data['RGB'].to(device)
        _sparse = data['SPARSE800'].to(device)
        _org_rgb = data['ORG_RGB'].to(device)
        _dense = data['DENSE800'].to(device)
        _mer = data['RADAR800'].to(device)
        _file_name = data['FILE_NAME']

        # org_sparse = _sparse_depth
        torch.cuda.synchronize()
        data_time = time.time() - end
        end = time.time()
        
        output_shape = _mer.shape
        all_pred = torch.zeros_like(_mer)
        counts = torch.zeros_like(_mer)
        
        cut = 3
        cut_shift = int((800-384)/(cut-1))
        
        with torch.no_grad():    
            for j in range(cut):
                h0 = 0
                h1 = 384
                w0 = int(0 + j*cut_shift)
                w1 = w0 + 384
                if w1 > 800:
                    w0 = 800 - 384
                    w1 = 800
                
                _input_rgb = _rgb[:,:,h0:h1, w0:w1]
                _input_mer = _mer[:,:,h0:h1, w0:w1]
                
                if modality == 'single':
                    _pred = model(_input_rgb)
                else:
                    _pred = model(_input_rgb, _input_mer)
                
                all_pred[:,:,h0:h1, w0:w1] = all_pred[:,:,h0:h1, w0:w1] + _pred
                counts[:,:,h0:h1, w0:w1] = counts[:,:,h0:h1, w0:w1] + 1.0
                
        final_pred = all_pred/counts

        torch.cuda.synchronize()
        gpu_time = time.time() - end
        
        s_abs_rel, s_sq_rel, s_rmse, s_rmse_log, s_a1, s_a2, s_a3 = compute_errors(_sparse, final_pred.to(device))
        d_abs_rel, d_sq_rel, d_rmse, d_rmse_log, d_a1, d_a2, d_a3 = compute_errors(_dense, final_pred.to(device))
        
        
        # measure accuracy and record loss
        result80_sparse = Result()
        result80_sparse.evaluate(final_pred, _sparse.data, cap=80)
        avg80_sparse.update(result80_sparse, gpu_time, data_time, _rgb.size(0))
        
        result50_sparse = Result()
        result50_sparse.evaluate(final_pred, _sparse.data, cap=50)
        avg50_sparse.update(result50_sparse, gpu_time, data_time, _rgb.size(0))
 
        end = time.time()
    
        # pred_result = utils.colored_depthmap(np.squeeze(final_pred.data.cpu().numpy()))
        _file_name = os.path.join(output_dir,'{}.png'.format(_file_name))
        img_merge = utils.merge_RGBandRADAR(_org_rgb, final_pred)
        utils.save_image(img_merge, _file_name)
        print('save prediction at {}'.format(_file_name))    

        # update progress bar and show loss
        evalbar.set_postfix(ORD_LOSS='||DENSE||RMSE={:.2f},delta={:.2f}/{:.2f}|||SPARSE||RMSE={:.2f},delta={:.2f}/{:.2f}|'.format(d_rmse,d_a1,d_a2,s_rmse,s_a1,s_a2))
        evalbar.update(1)
        i = i+1

    write_result(file, avg80_sparse, ' FINAL pred with CAP=80 ') 
    write_result(file, avg50_sparse, ' FINAL pred with CAP=50 ')

        
import argparse
parser = argparse.ArgumentParser(description='inference')
parser.add_argument('--model', type=str,
                    help="which model to use",
                    choices=['DPT', 'RCDPT', 'DPT_early', 'DPT_late'],
                    default='RCDPT')
parser.add_argument('--model_dir', help='ckpt model dir')
parser.add_argument('--ckpt', help='ckpt')
args = parser.parse_args()

# set arguments
BATCH_SIZE = 1
WORKERS = 3
SEED = 1984



# set random seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# load model
print('GPU number: {}'.format(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

CHECKPOINT = '{}/{}'.format(args.model_dir, args.ckpt)
print('load ckpt: {}'.format(CHECKPOINT))
pretrained_dict = torch.load(CHECKPOINT,map_location="cpu")
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict['model_state_dict'].items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_dict) 
model.load_state_dict(pretrained_dict, strict=False)

# create output dir,
output_dir = '{}/evaluation'.format(args.model_dir)
print('OUTPUT_DIR = {}'.format(output_dir))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
val_loader = nusc_loader_mer_eval.init_data_loader(mode='val', shuffle=False, batch_size=BATCH_SIZE, num_workers=WORKERS)

# if GPU number > 1, then use multiple GPUs
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)


# dump evalutaion result
file = open("{}/val_result.txt".format(output_dir),"a") 
file.write('{}\n'.format(args.model_dir))

validation(device, val_loader, model, output_dir, file, modality)

file.close()
print('DONE evaluateing {}'.format(args.model_dir))


