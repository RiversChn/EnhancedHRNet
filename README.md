# yolo weight
weights下要下载对应的预训练的权重  
在./models/detectors下   
git clone https://github.com/eriklindernoren/PyTorch-YOLOv3  
把文件名改为yolo后，在yolo文件下
安装环境文件  
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ -r requirements.txt  
cd weights/
下载weight  :bash download_weights.sh  
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
# 训练HRNet
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

# 训练
python scripts/train_coco.py --help  
需要传入list参数retain_block,代表多分支结构各个层的Block数量，retain_block通过卷积贡献度模块计算得出
