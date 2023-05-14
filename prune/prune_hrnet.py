import argparse
from collections import OrderedDict
import os
import re
import copy
from datetime import datetime
import torch
import torch.nn as nn
from models_.hrnet import HRNet
from datasets.COCO import COCODataset
from training.COCO import COCOTrain
from prune.Classifier import Classifier
import ast
import torch
import torch.nn as nn
import tqdm
from losses.loss import JointsMSELoss, JointsOHKMMSELoss
from torch.optim import SGD, Adam
from torch.utils.data.dataloader import DataLoader

def getData():
    coco_root_path = "../datasets/COCO"
    image_resolution = '(384, 288)'
    coco_bbox_path = None
    image_resolution = ast.literal_eval(image_resolution)


    ds_train = COCODataset(
        root_path=coco_root_path, data_version="train2017", is_train=True, use_gt_bboxes=True, bbox_path="",
        image_width=image_resolution[1], image_height=image_resolution[0], color_rgb=True,
    )

    ds_val = COCODataset(
        root_path=coco_root_path, data_version="val2017", is_train=False, use_gt_bboxes=(coco_bbox_path is None),
        bbox_path=coco_bbox_path, image_width=image_resolution[1], image_height=image_resolution[0], color_rgb=True,
    )
    return ds_train,ds_val


def find_ckpt(logdir):
    ckpt_names = []
    for filename in os.listdir(logdir):
        if filename == 'checkpoint_best_acc.pth':
            ckpt_names.append(filename)
    assert len(ckpt_names) == 1
    checkpoint = torch.load(os.path.join(logdir, ckpt_names[0]))
    return checkpoint



def main(model_c = 48,model_nof_joints=17,     model_bn_momentum=0.1,):
    print("ok")
    classifier = Classifier(32, 16, 10, True).cuda()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model = HRNet(c=model_c, nof_joints=model_nof_joints,
                           bn_momentum=model_bn_momentum).to(device)
    ckpt_read_dir =  os.path.join('../scripts/logs','20230316_0354')
    # pruned_channels = OrderedDict()
    checkpoint = find_ckpt(ckpt_read_dir)

    old_state_dict = checkpoint['model']
    model.load_state_dict(old_state_dict)
    teacher_model = model
    ds_train,ds_val = getData()
    dl_train = DataLoader(ds_train, batch_size=16, shuffle=True,
                               num_workers=1, drop_last=True)
    # print(OrderedDict(model._layers.named_modules()).keys())
    # import IPython
    # IPython.embed()
    train_per_block(model,ds_train,ds_val,3)
    print("ok")
    print("ok")


def train_per_block(model,  trainloader, testloader, epochs):



    device = torch.device('cuda:0')

    classifier =  nn.Conv2d(48, 17, kernel_size=(1, 1), stride=(1, 1))
    optim = Adam(model.parameters(), lr=0.001, weight_decay=0)
    loss_fn = JointsMSELoss().to(device)
    mean_loss_train = 0
    mean_acc_train = 0
    for _ in range(epochs):


        for image, target, target_weight, joints_data in trainloader:
            image = image.to(device)
            # target = target.to(device)
            # target_weight = target_weight.to(device)
            # _,endpoints = model(image,True)
            y, endpoints = model(torch.ones(1, 3, 384, 288).to(device), True)

            EP = endpoints[0][0]
            classifier.train()
            optim.zero_grad()

            output =  classifier(EP)

            loss = loss_fn(output, target, target_weight)

            loss.backward()

            optim.step()


    return []





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", "-n",
                        help="experiment name. A folder with this name will be created in the log_path.",
                        type=str, default=str(datetime.now().strftime("%Y%m%d_%H%M")))
    parser.add_argument("--epochs", "-e", help="number of epochs", type=int, default=1)
    parser.add_argument("--batch_size", "-b", help="batch size", type=int, default=4)
    parser.add_argument("--num_workers", "-w", help="number of DataLoader workers", type=int, default=4)
    parser.add_argument("--lr", "-l", help="initial learning rate", type=float, default=0.001)
    parser.add_argument("--disable_lr_decay", help="disable learning rate decay", action="store_true")
    parser.add_argument("--lr_decay_steps", help="learning rate decay steps", type=str, default="(170, 200)")
    parser.add_argument("--lr_decay_gamma", help="learning rate decay gamma", type=float, default=0.1)
    parser.add_argument("--optimizer", "-o", help="optimizer name. Currently, only `SGD` and `Adam` are supported.",
                        type=str, default='Adam')
    parser.add_argument("--weight_decay", help="weight decay", type=float, default=0.)
    parser.add_argument("--momentum", "-m", help="momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", help="enable nesterov", action="store_true")
    parser.add_argument("--pretrained_weight_path", "-p",
                        help="pre-trained weight path. Weights will be loaded before training starts.",
                        type=str, default=None)
    parser.add_argument("--checkpoint_path", "-c",
                        help="previous checkpoint path. Checkpoint will be loaded before training starts. It includes "
                             "the model, the optimizer, the epoch, and other parameters.",
                        type=str, default=None)
    parser.add_argument("--log_path", help="log path. tensorboard logs and checkpoints will be saved here.",
                        type=str, default='./logs')
    parser.add_argument("--disable_tensorboard_log", "-u", help="disable tensorboard logging", action="store_true")
    parser.add_argument("--model_c", help="HRNet c parameter", type=int, default=48)
    parser.add_argument("--model_nof_joints", help="HRNet nof_joints parameter", type=int, default=17)
    parser.add_argument("--model_bn_momentum", help="HRNet bn_momentum parameter", type=float, default=0.1)
    parser.add_argument("--disable_flip_test_images", help="disable image flip during evaluation", action="store_true")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')
    parser.add_argument("--coco_root_path", help="COCO dataset root path", type=str, default="../datasets/COCO")
    parser.add_argument("--coco_bbox_path", help="path of detected bboxes to use during evaluation",
                        type=str, default=None)
    parser.add_argument("--seed", "-s", help="seed", type=int, default=1)
    parser.add_argument("--device", "-d", help="device", type=str, default=None)
    args = parser.parse_args()

    main()
