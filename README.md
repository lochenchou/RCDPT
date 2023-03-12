# RCDPT
Radar-Camera Fusion Dense Prediction Transformer

Official implementation of "RCDPT: Radar-Camera Fusion Dense Prediction Transformer" (https://arxiv.org/abs/2211.02432), an accepted paper of ICASSP2023.

## Dependency

Please check `Dockerfile` for environment settings and python packages

Or you could directly use the pre-built docker image with tag 'lochenchou/det:mde' from docker hub.

## Usage

### Generating Multi-channel Enhanced Radar 

For generating Multi-channel Enhanced Radar (MER), which is the radar format used in the experiment, please follow the instructions of RC-PDA (https://github.com/longyunf/rc-pda).

### Generating Sparse Lidar depthmap.

Please follow the steps in `gen_interpolation.py` in DORN_radar repo (https://github.com/lochenchou/DORN_radar) to generate 5 frames sparse lidar as the golden ground truth to evaluate with during the evaluation step.

### Train baseline model and proposed model on nuScenes

Directly call 'train.py' with dataset paths for training the baseline model and the proposed model on nuScenes.


## Citation

If you find this work useful in your research, please consider citing:
```
@inproceedings{RCDPT,
  author={Lo, Chen-Chou and Vandewalle, Patrick},
  booktitle={Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2023)}, 
  title={RCDPT: Radar-Camera fusion Dense Prediction Transformer}, 
  year={2023},
}
```



