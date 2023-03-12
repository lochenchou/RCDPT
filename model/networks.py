import numpy as np
import torch
import torch.nn as nn
import timm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch.nn.functional as F

from model.Reassemble import Reassemble, Reassemble_multi
from model.Fusion import Fusion, Fusion_multi
from model.Head import HeadDepth

torch.manual_seed(0)

class DPT(nn.Module):
    def __init__(self,
                 image_size         = (3, 384, 384),
                 patch_size         = 16,
                 emb_dim            = 768,
                 resample_dim       = 256,
                 read               = 'projection',
                 num_layers_encoder = 24,
                 hooks              = [2, 5, 8, 11],
                 reassemble_s       = [4, 8, 16, 32],
                 transformer_dropout= 0,
                 model_timm         = "vit_base_patch16_384"):
        """
        Dense Prediction Transformer
        from Vision Transformers for Dense Prediction
        https://arxiv.org/abs/2103.13413
        image_size : (c, h, w)
        patch_size : *a square*
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        """
        super().__init__()
        
        # Vision Transformer backbone
        self.transformer_encoders = timm.create_model(model_timm, pretrained=True)

        # Hooks
        self.activation = {}
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)

        # Reassembles & Fusion layers
        self.reassembles = []
        self.fusions = []
        for s in reassemble_s:
            self.reassembles.append(Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim))
            self.fusions.append(Fusion(resample_dim))
        self.reassembles = nn.ModuleList(self.reassembles)
        self.fusions = nn.ModuleList(self.fusions)

        # Depth head
        self.head_depth = HeadDepth(resample_dim)
            
    def forward(self, img):
        t = self.transformer_encoders(img)  
        previous_stage = None
        
        for i in np.arange(len(self.fusions)-1, -1, -1):
            hook_to_take = 't'+str(self.hooks[i])
            activation_result = self.activation[hook_to_take]
            reassemble_result = self.reassembles[i](activation_result)
            fusion_result = self.fusions[i](reassemble_result, previous_stage)
            previous_stage = fusion_result

        out_depth = self.head_depth(previous_stage)

        return out_depth

    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        for h in hooks:
            self.transformer_encoders.blocks[h].register_forward_hook(get_activation('t'+str(h)))
            
                 
class RCDPT(nn.Module):
    def __init__(self,
                 image_size         = (3, 384, 384),
                 patch_size         = 16,
                 emb_dim            = 768,
                 resample_dim       = 256,
                 read               = 'projection',
                 num_layers_encoder = 24,
                 hooks              = [2, 5, 8, 11],
                 reassemble_s       = [4, 8, 16, 32],
                 transformer_dropout= 0,
                 model_timm         = "vit_base_patch16_384"):
        """
        RCDPT: Radar-Camera fusion Dense Prediction Transformer
        https://arxiv.org/abs/2211.02432
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)
    
        # Vision Transformer backbone 
        self.transformer_encoders_RGB = timm.create_model(model_timm, pretrained=True)
        self.transformer_encoders_radar = timm.create_model(model_timm, pretrained=True)

        # Hooks
        self.activation_RGB = {}
        self.activation_radar = {}
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)
                 
        # Reassembles & Fusion layers
        self.reassembles = []
        self.fusions = []
        for s in reassemble_s:
            self.reassembles.append(Reassemble_multi(image_size, read, patch_size, s, emb_dim, resample_dim))
            self.fusions.append(Fusion(resample_dim))
        self.reassembles = nn.ModuleList(self.reassembles)
        self.fusions = nn.ModuleList(self.fusions)

        # Depth head
        self.head_depth = HeadDepth(resample_dim)

    def forward(self, img, radar):
        x = self.conv1(radar)
        x = self.relu(x)
        t_rgb = self.transformer_encoders_RGB(img)
        t_radar = self.transformer_encoders_radar(x)
        previous_stage = None
                 
        for i in np.arange(len(self.fusions)-1, -1, -1):
            hook_to_take = 't'+str(self.hooks[i])
            activation_result_RGB = self.activation_RGB[hook_to_take]
            activation_result_radar = self.activation_radar[hook_to_take]
            reassemble_result = self.reassembles[i](activation_result_RGB, activation_result_radar)
            fusion_result = self.fusions[i](reassemble_result, previous_stage)
            previous_stage = fusion_result

        out_depth = self.head_depth(previous_stage)
        
        return out_depth

    def _get_layers_from_hooks(self, hooks):
        def get_activation_RGB(name):
            def hook(model, input, output):
                self.activation_RGB[name] = output
            return hook
        def get_activation_radar(name):
            def hook(model, input, output):
                self.activation_radar[name] = output
            return hook
        for h in hooks:
            self.transformer_encoders_RGB.blocks[h].register_forward_hook(get_activation_RGB('t'+str(h)))
            self.transformer_encoders_radar.blocks[h].register_forward_hook(get_activation_radar('t'+str(h)))
 

                 
class DPT_early(nn.Module):
    def __init__(self,
                 image_size         = (3, 384, 384),
                 patch_size         = 16,
                 emb_dim            = 768,
                 resample_dim       = 256,
                 read               = 'projection',
                 num_layers_encoder = 24,
                 hooks              = [2, 5, 8, 11],
                 reassemble_s       = [4, 8, 16, 32],
                 transformer_dropout= 0,
                 model_timm         = "vit_base_patch16_384"):
        super().__init__()
        
        self.conv1 = nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=False)
        
        self.transformer_encoders = timm.create_model(model_timm, pretrained=True)

        # Hooks
        self.activation = {}
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)

        # Reassembles & Fusion
        self.reassembles = []
        self.reassembles_radar = []
        self.fusions = []
        for s in reassemble_s:
            self.reassembles.append(Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim))
            self.fusions.append(Fusion(resample_dim))
        self.reassembles = nn.ModuleList(self.reassembles)
        self.fusions = nn.ModuleList(self.fusions)

        # Depth head
        self.head_depth = HeadDepth(resample_dim)

    def forward(self, img, radar):
        
        x = torch.cat((img, radar), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        t = self.transformer_encoders(x)
        
        previous_stage = None
        for i in np.arange(len(self.fusions)-1, -1, -1):
            hook_to_take = 't'+str(self.hooks[i])
            activation_result = self.activation[hook_to_take]
            reassemble_result = self.reassembles[i](activation_result)
            fusion_result = self.fusions[i](reassemble_result, previous_stage)
            previous_stage = fusion_result
  
        out_depth = self.head_depth(previous_stage)

        return out_depth

    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        for h in hooks:
            self.transformer_encoders.blocks[h].register_forward_hook(get_activation('t'+str(h)))
                   

class DPT_late(nn.Module):
    def __init__(self,
                 image_size         = (3, 384, 384),
                 patch_size         = 16,
                 emb_dim            = 768,
                 resample_dim       = 256,
                 read               = 'projection',
                 num_layers_encoder = 24,
                 hooks              = [2, 5, 8, 11],
                 reassemble_s       = [4, 8, 16, 32],
                 transformer_dropout= 0,
                 model_timm         = "vit_base_patch16_384"):    
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)
        
        self.transformer_encoders_RGB = timm.create_model(model_timm, pretrained=True)
        self.transformer_encoders_radar = timm.create_model(model_timm, pretrained=True)

        # Hooks
        self.activation_RGB = {}
        self.activation_radar = {}
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)
                 
        # Reassembles & Fusion
        self.reassembles_RGB = []
        self.fusions_RGB = []
        # for radar only
        self.reassembles_radar = []
        self.fusions_radar = []
        
        for s in reassemble_s:
            self.reassembles_RGB.append(Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim))
            self.fusions_RGB.append(Fusion(resample_dim))
            
            self.reassembles_radar.append(Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim))
            self.fusions_radar.append(Fusion(resample_dim))
            
        self.reassembles_RGB = nn.ModuleList(self.reassembles_RGB)
        self.fusions_RGB = nn.ModuleList(self.fusions_RGB)
        self.reassembles_radar = nn.ModuleList(self.reassembles_radar)
        self.fusions_radar = nn.ModuleList(self.fusions_radar)

        self.head_depth = HeadDepth(resample_dim+resample_dim)

    def forward(self, img, radar):

        x = self.conv1(radar)
        x = self.relu(x)

        t_rgb = self.transformer_encoders_RGB(img)
        t_radar = self.transformer_encoders_radar(x)
        
        previous_stage_RGB = None
        previous_stage_radar = None
        
        for i in np.arange(len(self.fusions_RGB)-1, -1, -1):
            hook_to_take = 't'+str(self.hooks[i])

            activation_result_RGB = self.activation_RGB[hook_to_take]
            activation_result_radar = self.activation_radar[hook_to_take]
            
            reassemble_result_RGB = self.reassembles_RGB[i](activation_result_RGB)
            fusion_result_RGB = self.fusions_RGB[i](reassemble_result_RGB, previous_stage_RGB)
            
            reassemble_radar = self.reassembles_radar[i](activation_result_radar)
            fusion_radar = self.fusions_radar[i](reassemble_radar, previous_stage_radar)
            
            previous_stage_RGB = fusion_result_RGB
            previous_stage_radar = fusion_radar
        
        previous_stage_merge = torch.cat((previous_stage_RGB, previous_stage_radar), dim=1)
        out_depth = self.head_depth(previous_stage_merge) 

        return out_depth
    
    
    def _get_layers_from_hooks(self, hooks):
        def get_activation_RGB(name):
            def hook(model, input, output):
                self.activation_RGB[name] = output
            return hook
        def get_activation_radar(name):
            def hook(model, input, output):
                self.activation_radar[name] = output
            return hook
        for h in hooks:
            self.transformer_encoders_RGB.blocks[h].register_forward_hook(get_activation_RGB('t'+str(h)))
            self.transformer_encoders_radar.blocks[h].register_forward_hook(get_activation_radar('t'+str(h)))                 

                 
                 