import os
import time
import torch
import numpy as np
import utils
from tqdm import tqdm
from metrics import AverageMeter, Result, compute_errors
import torch.nn.functional as F
from loss import MaskedL1Loss, SmoothnessLoss


def train_one_epoch(device, train_loader, model, output_dir, optimizer, epoch, logger, modality, print_freq):
    
    avg80_sparse = AverageMeter()
    avg80_dense = AverageMeter()
    L1 = MaskedL1Loss()
    Smooth = SmoothnessLoss()
    
    model.train()

    iter_per_epoch = len(train_loader)
    trainbar = tqdm(total=iter_per_epoch, position=0, leave=True)
    end = time.time()
  
    for i, data in enumerate(train_loader):
        _rgb384 = data['RGB384'].to(device)
        _sparse384 = data['SPARSE384'].to(device)
        _org_rgb384 = data['ORG_RGB384'].to(device)
        _dense384 = data['DENSE384'].to(device)
        _mer384 = data['RADAR384'].to(device)
        
        torch.cuda.synchronize()
        data_time = time.time() - end
                
        end = time.time()
        
        if modality == 'single':
            out_pred = model(_rgb384)
        else:
            out_pred = model(_rgb384,_mer384)
            
        final_loss = L1(out_pred, _sparse384)
        gradient_loss = Smooth(out_pred, _org_rgb384)
        _loss = final_loss + 0.1*gradient_loss
 
        optimizer.zero_grad()
        _loss.backward()  
        optimizer.step()
            
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # calculate metrices with ground truth sparse depth
        s_abs_rel, s_sq_rel, s_rmse, s_rmse_log, s_a1, s_a2, s_a3 = compute_errors(_sparse384, out_pred.to(device))
        d_abs_rel, d_sq_rel, d_rmse, d_rmse_log, d_a1, d_a2, d_a3 = compute_errors(_dense384, out_pred.to(device))
        
        result80_sparse = Result()
        result80_sparse.evaluate(out_pred, _sparse384.data, cap=80)
        avg80_sparse.update(result80_sparse, gpu_time, data_time, _rgb384.size(0))
        
        result80_dense = Result()
        result80_dense.evaluate(out_pred, _dense384.data, cap=80)
        avg80_dense.update(result80_dense, gpu_time, data_time, _rgb384.size(0))

        end = time.time()

        # update progress bar and show loss
        trainbar.set_postfix(LOSS='{:.2f}||EPOCH={}||DENSE||RMSE={:.2f},delta={:.2f}/{:.2f}|||SPARSE||RMSE={:.2f},delta={:.2f}/{:.2f}|'.format(_loss,epoch,d_rmse,d_a1,d_a2,s_rmse,s_a1,s_a2))
        trainbar.update(1)
        
        if (i + 1) % print_freq == 0:        
            print('SPARSE: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'RMSE_log={result.rmse_log:.3f}({average.rmse_log:.3f}) '
                  'AbsRel={result.absrel:.2f}({average.absrel:.2f}) '
                  'SqRel={result.squared_rel:.2f}({average.squared_rel:.2f}) '
                  'SILog={result.silog:.2f}({average.silog:.2f}) '
                  'iRMSE={result.irmse:.2f}({average.irmse:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
                  'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
                i + 1, len(train_loader), gpu_time=gpu_time, result=result80_sparse, average=avg80_sparse.average()))

            current_step = int(epoch*iter_per_epoch+i+1)
        
            img_merge = utils.batch_merge_into_row_radar2(_org_rgb384, _mer384.data, _sparse384.data, out_pred, out_pred)
            filename = os.path.join(output_dir,'step_{}.png'.format(current_step))
            utils.save_image(img_merge, filename)

            logger.add_scalar('TRAIN/SPARSE_RMSE', avg80_sparse.average().rmse, current_step)
            logger.add_scalar('TRAIN/SPARSE_RMSE_log', avg80_sparse.average().rmse_log, current_step)
            logger.add_scalar('TRAIN/SPARSE_iRMSE', avg80_sparse.average().irmse, current_step)
            logger.add_scalar('TRAIN/SPARSE_SILog', avg80_sparse.average().silog, current_step)
            logger.add_scalar('TRAIN/SPARSE_AbsRel', avg80_sparse.average().absrel, current_step)
            logger.add_scalar('TRAIN/SPARSE_SqRel', avg80_sparse.average().squared_rel, current_step)
            logger.add_scalar('TRAIN/SPARSE_Delta1', avg80_sparse.average().delta1, current_step)
            logger.add_scalar('TRAIN/SPARSE_Delta2', avg80_sparse.average().delta2, current_step)
            logger.add_scalar('TRAIN/SPARSE_Delta3', avg80_sparse.average().delta3, current_step)
            
            logger.add_scalar('TRAIN/DENSE_RMSE', avg80_dense.average().rmse, current_step)
            logger.add_scalar('TRAIN/DENSE_RMSE_log', avg80_dense.average().rmse_log, current_step)
            logger.add_scalar('TRAIN/DENSE_iRMSE', avg80_dense.average().irmse, current_step)
            logger.add_scalar('TRAIN/DENSE_SILog', avg80_dense.average().silog, current_step)
            logger.add_scalar('TRAIN/DENSE_AbsRel', avg80_dense.average().absrel, current_step)
            logger.add_scalar('TRAIN/DENSE_SqRel', avg80_dense.average().squared_rel, current_step)
            logger.add_scalar('TRAIN/DENSE_Delta1', avg80_dense.average().delta1, current_step)
            logger.add_scalar('TRAIN/DENSE_Delta2', avg80_dense.average().delta2, current_step)
            logger.add_scalar('TRAIN/DENSE_Delta3', avg80_dense.average().delta3, current_step)
            
            # reset average meter
            result80_sparse = Result()
            avg80_sparse = AverageMeter()   
            result80_dense = Result()
            avg80_dense = AverageMeter()  

            
def validation(device, data_loader, model, output_dir, epoch, logger, modality):
    avg80_sparse = AverageMeter()
    avg80_dense = AverageMeter()
    L1 = MaskedL1Loss()
    
    model.eval()
    
    end = time.time()
    skip =int(len(data_loader)/10)
    img_list = []
    
    evalbar = tqdm(total=len(data_loader))
    
    for i, data in enumerate(data_loader):
        _rgb384 = data['RGB384'].to(device)
        _sparse384 = data['SPARSE384'].to(device)
        _org_rgb384 = data['ORG_RGB384'].to(device)
        _dense384 = data['DENSE384'].to(device)
        _mer384 = data['RADAR384'].to(device)
        
        torch.cuda.synchronize()
        data_time = time.time() - end
        
        # compute output
        end = time.time()
        with torch.no_grad():
            if modality == 'single':
                out_pred = model(_rgb384)
            else:
                out_pred = model(_rgb384,_mer384)
                
            _loss = L1(out_pred, _sparse384)
            
        torch.cuda.synchronize()
        gpu_time = time.time() - end
        
        s_abs_rel, s_sq_rel, s_rmse, s_rmse_log, s_a1, s_a2, s_a3 = compute_errors(_sparse384, out_pred.to(device))
        d_abs_rel, d_sq_rel, d_rmse, d_rmse_log, d_a1, d_a2, d_a3 = compute_errors(_dense384, out_pred.to(device))
        
        # measure accuracy and record loss
        result80_sparse = Result()
        result80_sparse.evaluate(out_pred, _sparse384.data, cap=80)
        avg80_sparse.update(result80_sparse, gpu_time, data_time, _rgb384.size(0))
        
        result80_dense = Result()
        result80_dense.evaluate(out_pred, _dense384.data, cap=80)
        avg80_dense.update(result80_dense, gpu_time, data_time, _rgb384.size(0))

        end = time.time()
        
        # save images for visualization 
        if i == 0:
            img_merge = utils.merge_into_row_with_radar2(_org_rgb384, _mer384.data, _sparse384.data, out_pred, out_pred)
        elif (i < 8 * skip) and (i % skip == 0):
            row = utils.merge_into_row_with_radar2(_org_rgb384, _mer384.data, _sparse384.data, out_pred, out_pred)
            img_merge = utils.add_row(img_merge, row)
        elif i == 8 * skip:
            filename = os.path.join(output_dir,'eval_{}.png'.format(int(epoch)))
            print('save validation figures at {}'.format(filename))
            utils.save_image(img_merge, filename)

        # update progress bar and show loss
        evalbar.set_postfix(LOSS='{:.2f}||AUX||RMSE={:.2f},delta={:.2f}/{:.2f}|||FINAL||RMSE={:.2f},delta={:.2f}/{:.2f}|'.format(_loss,d_rmse,d_a1,d_a2,s_rmse,s_a1,s_a2))
        evalbar.update(1)

        i = i+1

    print('\n**** EVALUATE WITH SPARSE DEPTH ****\n'
          '\n**** CAP=80 ****\n'
          'RMSE={average.rmse:.3f}\n'
          'RMSE_log={average.rmse_log:.3f}\n'
          'AbsRel={average.absrel:.3f}\n'
          'SqRel={average.squared_rel:.3f}\n'
          'SILog={average.silog:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'Delta2={average.delta2:.3f}\n'
          'Delta3={average.delta3:.3f}\n'
          'iRMSE={average.irmse:.3f}\n'
          'iMAE={average.imae:.3f}\n'
          't_GPU={average.gpu_time:.3f}\n'.format(
        average=avg80_sparse.average()))
    
    logger.add_scalar('VAL_CAP80/SPARSE_RMSE', avg80_sparse.average().rmse, epoch)
    logger.add_scalar('VAL_CAP80/SPARSE_RMSE_log', avg80_sparse.average().rmse_log, epoch)
    logger.add_scalar('VAL_CAP80/SPARSE_iRMSE', avg80_sparse.average().irmse, epoch)
    logger.add_scalar('VAL_CAP80/SPARSE_SILog', avg80_sparse.average().silog, epoch)
    logger.add_scalar('VAL_CAP80/SPARSE_AbsRel', avg80_sparse.average().absrel, epoch)
    logger.add_scalar('VAL_CAP80/SPARSE_SqRel', avg80_sparse.average().squared_rel, epoch)
    logger.add_scalar('VAL_CAP80/SPARSE_Delta1', avg80_sparse.average().delta1, epoch)
    logger.add_scalar('VAL_CAP80/SPARSE_Delta2', avg80_sparse.average().delta2, epoch)
    logger.add_scalar('VAL_CAP80/SPARSE_Delta3', avg80_sparse.average().delta3, epoch)
    
    logger.add_scalar('VAL_CAP80/DENSE_RMSE', avg80_dense.average().rmse, epoch)
    logger.add_scalar('VAL_CAP80/DENSE_RMSE_log', avg80_dense.average().rmse_log, epoch)
    logger.add_scalar('VAL_CAP80/DENSE_iRMSE', avg80_dense.average().irmse, epoch)
    logger.add_scalar('VAL_CAP80/DENSE_SILog', avg80_dense.average().silog, epoch)
    logger.add_scalar('VAL_CAP80/DENSE_AbsRel', avg80_dense.average().absrel, epoch)
    logger.add_scalar('VAL_CAP80/DENSE_SqRel', avg80_dense.average().squared_rel, epoch)
    logger.add_scalar('VAL_CAP80/DENSE_Delta1', avg80_dense.average().delta1, epoch)
    logger.add_scalar('VAL_CAP80/DENSE_Delta2', avg80_dense.average().delta2, epoch)
    logger.add_scalar('VAL_CAP80/DENSE_Delta3', avg80_dense.average().delta3, epoch)
 