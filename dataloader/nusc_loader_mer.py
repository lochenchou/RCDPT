import torch
from torch.utils import data
import numpy as np
import os
from os.path import join
import PIL
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
import h5py
from nuscenes import NuScenes
import random
import cv2
random.seed(1984)



def getDescription(nusc, camera_token):
    data = nusc.get('sample_data', camera_token)
    sample = nusc.get('sample', data['sample_token'])
    scene = nusc.get('scene', sample['scene_token'])
    return scene['description'].lower()


def pil_loader(path, rgb=True):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if rgb:
            return img.convert('RGB')
        else:
            return img.convert('I')


def resize_depth(depth, _size=(450,800)):
    
    _depth = np.array(depth)
    
    if _depth.shape == _size:
        return _depth
    
    depth_w = _depth.shape[1]
    size_w = _size[1]
    _f = int(depth_w/size_w)
    
    re_depth = np.zeros(_size)
    pts = np.where(_depth!=0)
    
    
    re_depth[(pts[0][:]/_f).astype(np.int), (pts[1][:]/_f).astype(np.int)] = _depth[pts[0][:], pts[1][:]]
    
    return re_depth


def getFileName(nusc, sample_idx):
    cam_token = nusc.sample[sample_idx]['data']['CAM_FRONT']
    data = nusc.get('sample_data', cam_token)
    sample = nusc.get('sample', data['sample_token'])
    scene_token = sample['scene_token']
    scene_name = nusc.get('scene',scene_token)['name']
    sample_timestamp = sample['timestamp']
    file_name = '{}-{}'.format(scene_name, sample_timestamp)
    return file_name

      
def init_data_loader(mode='train', shuffle=True, batch_size=4, num_workers=4, sample_index=0, nusc=None):    
    args_dataset = {'path_data_file': '/datasets/nuscenes/prepared_data.h5',
                    'path_radar_file': '/datasets/nuscenes/mer_2_30_5_0.5.h5',
                    'mode': mode,
                    'sample_index': sample_index, 
                    'nusc': nusc,
                   }
    args_data_loader = {'batch_size': batch_size,
                       'shuffle': shuffle,
                       'num_workers': num_workers}
    dataset = Dataset(**args_dataset)    
    data_loader = torch.utils.data.DataLoader(dataset, **args_data_loader)
    
    return data_loader
    

class Dataset(data.Dataset):     
    def __init__(self, path_data_file, path_radar_file, mode, sample_index, nusc):               
        
        if mode != 'test':
            ROOT_PATH = '/datasets/nuscenes/v1.0-trainval'
            VERSION = 'v1.0-trainval'
        else:
            ROOT_PATH = '/datasets/nuscenes/v1.0-test'
            VERSION = 'v1.0-test'            
        
        if nusc == None:
            self.nusc = NuScenes(version=VERSION, dataroot=ROOT_PATH, verbose=True)
        else:
            self.nusc = nusc
        
        self.mode = mode
        self.sample_index = sample_index
        self.data = h5py.File(path_data_file, 'r')[mode] 
        self.data_radar = h5py.File(path_radar_file, 'r')[mode] 
        self.cap = 80
        self.indices = self.data['indices']
        self.num_sample = len(self.indices) - 10
        print('num_sample = {}'.format(self.num_sample))
        
        print('Dataset initialization DONE')
                           
    def __len__(self):
        'Denotes the total number of samples'
        return self.num_sample
        
    def __getitem__(self, idx):
        'Generate one sample of data'
        
        sample_idx = self.indices[idx]
        cam_token = self.nusc.sample[sample_idx]['data']['CAM_FRONT']
        cam = self.nusc.get('sample_data', cam_token)
        sparse5_path = os.path.join(self.nusc.dataroot, 'samples', 'SPARSE_5_FRONT', cam['filename'].split('/')[-1].replace('jpg','png'))
        dense_path = os.path.join(self.nusc.dataroot, 'samples', 'DENSE_FRONT', cam['filename'].split('/')[-1].replace('jpg','png'))
        RGB_path = os.path.join(self.nusc.dataroot, cam['filename'])
        desc = getDescription(self.nusc, cam_token)
        
        rgb = pil_loader(RGB_path, rgb=True)
        rgb800 = rgb.resize((800,450))    
        rgb800 = np.array(rgb800).astype(np.float32) / 255.
        rgb800=rgb800[26::,:]  # 424x800

        sparse = pil_loader(sparse5_path, rgb=False)
        sparse = np.array(sparse).astype(np.float32) / 256. 
        sparse800 = resize_depth(sparse, (450,800))
        sparse800 = sparse800[26::,:]  # 424x800

        dense = pil_loader(dense_path, rgb=False)
        dense = np.array(dense).astype(np.float32) / 256.
        dense800 = resize_depth(dense, (450,800))
        dense800 = dense800[26::,:]  # 424x800  (384+40)
        

#         # get RGB & radar
        _mer400 = self.data_radar['radar'][idx,self.sample_index,:,:].astype('float32')/100             # centimeter to meter, 192x400
        _mer800 = cv2.resize(_mer400, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) # 384x800
        d_mer800 = np.zeros(shape=(384+40,800), dtype=np.float32) # 424x800
        d_mer800[40::,:] = _mer800

        _lidar400 = self.data['gt'][idx,:,:,0].astype('float32')            # (1,h,w) # 192x400
        _lidar800 = cv2.resize(_lidar400, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) # 384x800
        d_lidar800 = np.zeros(shape=(384+40,800), dtype=np.float32) # 424x800
        d_lidar800[40::,:] = _lidar800


        if self.mode == 'train':
            shifty800 = random.randint(0, 40)
            shiftx800 = random.randint(0, (800-384))
            
            rgb384 = rgb800[shifty800:shifty800+384,shiftx800:shiftx800+384,:]
            sparse384 = sparse800[shifty800:shifty800+384,shiftx800:shiftx800+384]
            dense384 = dense800[shifty800:shifty800+384,shiftx800:shiftx800+384]
            d_lidar384 = d_lidar800[shifty800:shifty800+384,shiftx800:shiftx800+384]
            d_mer384 = d_mer800[shifty800:shifty800+384,shiftx800:shiftx800+384]
           
            do_flip = random.random()
            if do_flip > 0.5:
                rgb384 = (rgb384[:, ::-1, :]).copy()
                sparse384 = (sparse384[:, ::-1]).copy()
                dense384 = (dense384[:, ::-1]).copy()
                d_lidar384 = (d_lidar384[:, ::-1]).copy()
                d_mer384 = (d_mer384[:, ::-1]).copy()
        else:
            shiftx = int((800-384)/2)
            rgb384 = rgb800[40::,shiftx:shiftx+384]
            sparse384 = sparse800[40::,shiftx:shiftx+384]
            dense384 = dense800[40::,shiftx:shiftx+384]
            d_lidar384 = d_lidar800[40::,shiftx:shiftx+384]
            d_mer384 = d_mer800[40::,shiftx:shiftx+384]

        rgb384 = T.ToTensor()(rgb384)       
        norm_rgb384 = T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])(rgb384)
        d_lidar384 = T.ToTensor()(d_lidar384)
        sparse384 = T.ToTensor()(sparse384)
        dense384 = T.ToTensor()(dense384)
        d_mer384 = T.ToTensor()(d_mer384)
                    
        return {
                'DESC': desc,
                'ORG_RGB384': rgb384.float(),
                'RGB384': norm_rgb384.float(),
                'SPARSE384': sparse384.float(),
                'DENSE384': d_lidar384.float(),
                'RADAR384': d_mer384.float(),
               }

