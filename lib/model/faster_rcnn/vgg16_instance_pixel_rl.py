# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import model.faster_rcnn.DQN as DQN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn_instance_pixel_rl import _fasterRCNN
#from model.faster_rcnn.faster_rcnn_imgandpixellevel_gradcam  import _fasterRCNN
from model.utils.config import cfg

import pdb
def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)
class netD_pixel(nn.Module):
    def __init__(self,context=False):
        super(netD_pixel, self).__init__()
        #self.conv1 = conv1x1(512, 256)
        #self.bn1 = nn.BatchNorm2d(256)
        #self.conv2 = conv1x1(256, 128)
        #self.bn2 = nn.BatchNorm2d(128)
        #self.conv3 = conv1x1(128, 1)
        self.conv1 = nn.Conv2d(512, 512, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.context = context
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
      #normal_init(self.conv4, 0, 0.01)
    def forward(self, x):
        #print("x.shape:",x.shape)
        x = self.leaky_relu(self.conv1(x))
        #print("x.shape in conv1 in pixelD:",x.shape)
        x = self.leaky_relu(self.conv2(x))
        #print("x.shape in conv2 in pixelD:",x.shape)
        #x = self.leaky_relu(self.conv3(x))
        #print("x.shape in conv3 in pixelD:",x.shape)
        if self.context:
            feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
            #print("feat.shape",feat.shape)
            # feat = x
            x = F.sigmoid(self.conv3(x))
            #print("x.shape in conv4 in pixelD:",x.shape)
            return x, feat  # torch.cat((feat1,feat2),1)#F
        else:
            x = self.conv3(x)
            return F.sigmoid(x)#F.sigmoid(x)

class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False,ic=False,S_agent=False,T_agent=False,ts=0.5,tt=0.5,select_num=3,candidate_num=16):
    self.model_path = cfg.VGG_PATH
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.ic = ic
    self.S_agent = S_agent
    self.T_agent = T_agent
    self.ts = ts
    self.tt = tt
    self.candidate_num = candidate_num
    self.select_num = select_num

    _fasterRCNN.__init__(self, classes, class_agnostic,ic,S_agent,T_agent,ts,tt,select_num,candidate_num)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    #print(vgg.features)
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    #self.RCNN_base2 = nn.Sequential(*list(vgg.features._modules.values())[14:-1])
    #print(self.RCNN_base1)
    #print(self.RCNN_base2)
    #self.netD = netD(context=self.gc)
    self.netD_pixel = netD_pixel(context=self.ic)
    action_num = cfg.CANDIDATE_NUM
    print("action_num",action_num)
    state_size = cfg.CANDIDATE_NUM*4096
    self.current_model = DQN.DQN(action_num,state_size)
    self.target_model = DQN.DQN(action_num,state_size)
    self.replay_buffer = DQN.ReplayBuffer(cfg.REPLAY_MEMORY,state_size)

    self.current_model_T = DQN.DQN(action_num,state_size)
    self.target_model_T = DQN.DQN(action_num,state_size)
    self.replay_buffer_T = DQN.ReplayBuffer(cfg.REPLAY_MEMORY,state_size) 

    feat_d = 4096
    if self.ic:
        feat_d += 128
    #if self.gc:
    #    feat_d += 128
    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)
    if self.class_agnostic:
        self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
    else:
        self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)


  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

