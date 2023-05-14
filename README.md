weights下要下载对应的预训练的权重
在./models/detectors下

git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
把文件名改为yolo后，在yolo文件下

#安装环境文件
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ -r requirements.txt
cd weights/
#下载weight
bash download_weights.sh

回到HRNet文件夹下

git submodule update --init --recursive

运行测试Class usage
import cv2
import matplotlib.pyplot as plt
from SimpleHRNet import SimpleHRNet

model = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth")
image = cv2.imread("./test.JPG", cv2.IMREAD_COLOR) #这里的文件路径和文件名根据自己要识别的图片来
joints = model.predict(image)
print(joints)



会出现ffmpeg报错，或是cv2没有安装的报错
继续配置环境

apt-get -y --no-install-recommends update 
apt-get -y --no-install-recommends upgrade 
apt-get install -y --no-install-recommends build-essential cmake git libatlas-base-dev libprotobuf-dev apt-file
apt-get install -y --no-install-recommends libleveldb-dev libsnappy-dev libhdf5-serial-dev libboost-all-dev
apt-get install -y --no-install-recommends libgflags-dev libgoogle-glog-dev libviennacl-dev libcanberra-gtk-module
apt-get install -y --no-install-recommends libopencv-dev opencl-headers ocl-icd-opencl-dev protobuf-compiler liblmdb-dev
apt-get install -y --no-install-recommends pciutils python3-setuptools python3-dev python3-pip
apt-get update 
apt-file update
#我安装上opencv后不能使用，然后安装下面包就可以了
apt-file search libSM.so.6 
apt-get install libsm6 
apt-get -y --no-install-recommends update 
apt-get -y --no-install-recommends upgrade
#重新安装ffmpeg
pip uninstall ffmpeg-python
sudo add-apt-repository ppa:djcj/hybrid  
sudo apt-get update  
sudo apt-get install ffmpeg  
conda install ffmpeg-python # 我测试过用pip安装ffmepg,并不行。
pip install json_tricks

训练HRNet
安装cocoapi
git clone https://github.com/cocodataset/cocoapi.git 
cd cocoapi-master/PythonAPI
# 使用 make
make install
# 或是下面代码
python3 setup.py install --user

安装nms
在simple-HRNet文件夹下

cd misc
make
#或是
cd misc/nms
python setup_linux.py build_ext --inplace

但是这样还是会包找不到cpu_nms和gpu_nms包的问题
所以需要打开misc/nms/nms.py文件
改变下面代码，把报错的注释掉，并没用上

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

#from cpu_nms import cpu_nms
#from gpu_nms import gpu_nms


def py_nms_wrapper(thresh):
    def _nms(dets):
        return nms(dets, thresh)
    return _nms


#def cpu_nms_wrapper(thresh):
#    def _nms(dets):
#        return cpu_nms(dets, thresh)
#    return _nms


#def gpu_nms_wrapper(thresh, device_id):
#    def _nms(dets):
#        return gpu_nms(dets, thresh, device_id)
#    return _nms

.csdn.net/weixin_44061195/article/details/107015655

训练
python scripts/train_coco.py --help

需要传入list参数retain_block,代表多分支结构各个层的Block数量，retain_block通过卷积贡献度模块计算得出

#usage: train_coco.py [-h] [--exp_name EXP_NAME] [--epochs EPOCHS]
#                     [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
#                     [--lr LR] [--disable_lr_decay]
#                     [--lr_decay_steps LR_DECAY_STEPS]
#                     [--lr_decay_gamma LR_DECAY_GAMMA] [--optimizer OPTIMIZER]
#                     [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM]
#                     [--nesterov]
#                     [--pretrained_weight_path PRETRAINED_WEIGHT_PATH]
#                     [--checkpoint_path CHECKPOINT_PATH] [--log_path LOG_PATH]
#                     [--disable_tensorboard_log] [--model_c MODEL_C]
#                     [--model_nof_joints MODEL_NOF_JOINTS]
#                     [--model_bn_momentum MODEL_BN_MOMENTUM]
#                     [--disable_flip_test_images]
#                     [--image_resolution IMAGE_RESOLUTION]
#                     [--coco_root_path COCO_ROOT_PATH]
#                     [--coco_bbox_path COCO_BBOX_PATH] [--seed SEED]
#                     [--device DEVICE]

#optional arguments:
#  -h, --help            show this help message and exit
#  --exp_name EXP_NAME, -n EXP_NAME
#                        experiment name. A folder with this name will be
#                        created in the log_path.
#  --epochs EPOCHS, -e EPOCHS
#                        number of epochs
#  --batch_size BATCH_SIZE, -b BATCH_SIZE
#                        batch size
#  --num_workers NUM_WORKERS, -w NUM_WORKERS
#                        number of DataLoader workers
#  --lr LR, -l LR        initial learning rate
#  --disable_lr_decay    disable learning rate decay
#  --lr_decay_steps LR_DECAY_STEPS
#                        learning rate decay steps
#  --lr_decay_gamma LR_DECAY_GAMMA
#                        learning rate decay gamma
#  --optimizer OPTIMIZER, -o OPTIMIZER
#                        optimizer name. Currently, only `SGD` and `Adam` are
#                       supported.
# --weight_decay WEIGHT_DECAY
#                        weight decay
# --momentum MOMENTUM, -m MOMENTUM
#                        momentum
#  --nesterov            enable nesterov
#  --pretrained_weight_path PRETRAINED_WEIGHT_PATH, -p PRETRAINED_WEIGHT_PATH
#                       pre-trained weight path. Weights will be loaded before
#                       training starts.
#  --checkpoint_path CHECKPOINT_PATH, -c CHECKPOINT_PATH
#                        previous checkpoint path. Checkpoint will be loaded
#                        before training starts. It includes the model, the
#                        optimizer, the epoch, and other parameters.
#  --log_path LOG_PATH   log path. tensorboard logs and checkpoints will be
#                        saved here.
#  --disable_tensorboard_log, -u
#                        disable tensorboard logging
#  --model_c MODEL_C     HRNet c parameter
#  --model_nof_joints MODEL_NOF_JOINTS
#                        HRNet nof_joints parameter
#  --model_bn_momentum MODEL_BN_MOMENTUM
#                        HRNet bn_momentum parameter
#  --disable_flip_test_images
#                        disable image flip during evaluation
#  --image_resolution IMAGE_RESOLUTION, -r IMAGE_RESOLUTION
#                        image resolution
#  --coco_root_path COCO_ROOT_PATH
#                        COCO dataset root path
#  --coco_bbox_path COCO_BBOX_PATH
#                        path of detected bboxes to use during evaluation
#  --seed SEED, -s SEED  seed
#  --device DEVICE, -d DEVICE
#                        device

