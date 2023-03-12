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
        

def read_list(list_path):
    print('read list from {}'.format(list_path))
    _list = []
    with open(list_path) as f:
        for line in f:
            _list.append(line.split('\n')[0])
            
    print(len(_list))
    return _list


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
        RGB_path = os.path.join(self.nusc.dataroot, cam['filename'])
        desc = getDescription(self.nusc, cam_token)
        file_name = getFileName(self.nusc, sample_idx)
        
        rgb = pil_loader(RGB_path, rgb=True)
        rgb800 = rgb.resize((800,450))    
        rgb800 = np.array(rgb800).astype(np.float32) / 255. 
        rgb800=rgb800[66::,:]  # 800x384
        
        sparse = pil_loader(sparse5_path, rgb=False)
        sparse = np.array(sparse).astype(np.float32) / 256.
        sparse800 = resize_depth(sparse, (450,800))
        sparse800 = sparse800[66::,:]  # 800x384
        
        # get RGB & radar
        d_mer400 = self.data_radar['radar'][idx,self.sample_index,:,:].astype('float32')/100             # centimeter to meter, 400x192
        d_mer800 = cv2.resize(d_mer400, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) # 384x800
        
        d_lidar400 = self.data['gt'][idx,:,:,0].astype('float32')            # (1,h,w) # 400x192
        d_lidar800 = cv2.resize(d_lidar400, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) # 384x800
        
        rgb800 = T.ToTensor()(rgb800)       
        norm_rgb800 = T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])(rgb800)
 
        d_lidar800 = T.ToTensor()(d_lidar800)
        sparse800 = T.ToTensor()(sparse800)
        d_mer800 = T.ToTensor()(d_mer800)

                           
        return {
                'DESC': desc,
                'ORG_RGB': rgb800.float(), 
                'RGB': norm_rgb800.float(),

                'DENSE800': d_lidar800.float(),
                'SPARSE800': sparse800.float(),
                'RADAR800': d_mer800.float(), # 384x800
            
                'FILE_NAME': file_name,
               }

