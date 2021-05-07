import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import model.faster_rcnn.DQN as DQN
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
import math
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta,grad_reverse

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic,context,S_agent,T_agent,ts,tt,select_num,candidate_num):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.select_num = select_num
        self.candidate_num = candidate_num
        print("self.select_num: %d self.candidate_num: %d "%(self.select_num,self.candidate_num))
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.context = context
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.epsilon_by_epoch = lambda epoch_idx: cfg.epsilon_final + (cfg.epsilon_start - \
            cfg.epsilon_final) * math.exp(-1. * epoch_idx / cfg.epsilon_decay)
        self.iter_dqn = 0

        self.epsilon_by_epoch_T = lambda epoch_idx: cfg.epsilon_final + (cfg.epsilon_start - \
            cfg.epsilon_final) * math.exp(-1. * epoch_idx / cfg.epsilon_decay)
        self.iter_dqn_T = 0

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    def forward(self, im_data, im_info, gt_boxes, num_boxes,target=False,eta=1.0):

        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        lossQ = -1

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        # feed base feature map tp RPN to obtain rois'''
        #print("target is ",target)
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes,target)
        #print("rois.shape:",rois.shape)
        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training and not target:
            #print("source traning---------------------------")
            #print("batch_size:",batch_size)
            #print("gt_boxes.shape:",gt_boxes.shape)
            #print("num_boxes:",num_boxes.data)
            '''
            print(self.training)
            print(~target)
            print("use ground trubut bboxes for refining")'''
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            lossQ = -1

        rois = Variable(rois)
        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        #print("pooled_feat before _head_to_tail:",pooled_feat.shape)
        if self.context:
            d_instance, _ = self.netD_pixel(grad_reverse(pooled_feat, lambd=eta))
            #if target:
                #d_instance, _ = self.netD_pixel(grad_reverse(pooled_feat, lambd=eta))
                #return d_pixel#, diff
            d_score_total,feat = self.netD_pixel(pooled_feat.detach())
        else:
            d_score_total = self.netD_pixel(pooled_feat.detach())
            d_instance = self.netD_pixel(grad_reverse(pooled_feat, lambd=eta))
            #if target:
                #return d_pixel#,diff

        #d_score_total, _ = self.netD_pixel(pooled_feat.detach())
        #print("d_score_total.shape",d_score_total.shape)
        #print("pooled_feat.shape:",pooled_feat.shape)
        d_instance_q = d_instance.split(128,0)

        d_score_total_q = d_score_total.split(128,0)
        d_score_total_qs = []
        for img in range(batch_size):
            temp = torch.mean(d_score_total_q[img],dim=3)
            d_score_total_qs.append(torch.mean(temp,dim=2))

        #d_score_total = torch.mean(d_score_total,dim=3)
        #d_score_total = torch.mean(d_score_total,dim=2)
        pooled_feat = self._head_to_tail(pooled_feat)
        
        #print("pooled_feat.shape:",pooled_feat.shape)

        if self.training and self.S_agent:
            pooled_feat_s = pooled_feat.split(128,0)
            for img in range(batch_size):
                pooled_feat_d = pooled_feat_s[img]
                #print("------------------begain selecting in the source-----------------------")
                select_iter = int(pooled_feat_d.shape[0]/self.candidate_num)
                total_index = list(range(0,pooled_feat_d.shape[0]))
                np.random.shuffle(total_index)
                select_index = []
                for eposide in range(select_iter):
                    #print("#################################begain batch-%d-th the %d-th eposide##################################" % (img,eposide)) 
                    select_list = list(range(0,self.candidate_num))
                    batch_idx = total_index[eposide*self.candidate_num:(eposide+1)*self.candidate_num]
                    state = pooled_feat_d[batch_idx]
                    #print("state.shape:",state.shape)
                    d_score = d_score_total_qs[img][batch_idx]       
                    #print("d_score.shape:",d_score.shape)                
                    for it in range(self.select_num):
                        #print("#########begain the %d-th selection################" % (it))      
                        epsilon = self.epsilon_by_epoch(self.iter_dqn)
                        action_index = self.current_model.act(state,epsilon,select_list)
                        #print("action_index:",action_index)
                        #action_episode.append(action_index)
                        try:
                            select_list.remove(action_index)
                        except:
                            print("select_list:",select_list)
                            print("action_index:",action_index)
                            print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            continue
                        #print("the %d-th select, action_index is %d"%(it,action_index))
                        if d_score[action_index] > self.ts:
                            reward = -1
                        else:
                            reward = 1     
                        #print("reward:",reward)
                        next_state = torch.tensor(state)            
                        next_state[action_index] = torch.zeros(1,next_state.shape[1])  
                        if it==(self.select_num-1):
                            done = 1
                        else:
                            done = 0
                        self.replay_buffer.push(state,action_index,reward,next_state,done,select_list)
                        self.iter_dqn = self.iter_dqn+1
                        state = next_state
                    select_index = select_index + [batch_idx[i] for i in select_list]
                if len(self.replay_buffer)>cfg.BATCH_SIZE_DQN:
                    lossQ = DQN.compute_td_loss(self.current_model,self.target_model,self.replay_buffer,cfg.BATCH_SIZE_DQN)         
                if np.mod(self.iter_dqn,cfg.replace_target_iter)==0:
                    DQN.update_target(self.current_model,self.target_model)
                if img==0:
                    d_instance_refine = d_instance_q[img][select_index]
                else:
                    d_instance_refine = torch.cat((d_instance_refine,d_instance_q[img][select_index]),0)
        pooled_feat_original = torch.tensor(pooled_feat)
        if self.context:
            feat = feat.view(feat.size(0),-1)
            pooled_feat = torch.cat((feat, pooled_feat), 1)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic and not target:
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        #print("pooled_feat.shape in faster_rcnn_global_pixel_instance:",pooled_feat.shape)
        cls_score = self.RCNN_cls_score(pooled_feat)

        cls_prob = F.softmax(cls_score, 1)
        #print("cls_prob is ",cls_prob.shape)

        if self.training and target and self.T_agent:
            pooled_feat_t = pooled_feat_original.split(128,0)
            for img in range(batch_size):
                pooled_feat_d = pooled_feat_t[img]

                select_iter_T = int(pooled_feat_d.shape[0]/self.candidate_num)
                #print("select_iter_T:",select_iter_T)
                total_index_T = list(range(0,pooled_feat_d.shape[0]))
                np.random.shuffle(total_index_T)
                #print("gt_label:",gt_label)
                #print("total_index:",len(total_index))
                select_index_T = []
                for eposide_T in range(select_iter_T):
                    select_list_T = list(range(0,self.candidate_num))                
                    batch_idx_T = total_index_T[eposide_T*self.candidate_num:(eposide_T+1)*self.candidate_num]
                    state_T = pooled_feat_d[batch_idx_T]
                    d_score_T = d_score_total_qs[img][batch_idx_T]
                    #print("label_pre:",label_pre)
                    for it in range(self.select_num):
                        epsilon_T = self.epsilon_by_epoch_T(self.iter_dqn_T)
                        action_index_T = self.current_model_T.act(state_T,epsilon_T,select_list_T)
                        #select_list_T.remove(action_index_T)
                        try:
                            select_list_T.remove(action_index_T)
                        except:
                            print("select_list_T:",select_list_T)
                            print("action_index:",action_index_T)
                            print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            continue
                        #print("label_pre[action_index_T]:",label_pre[action_index_T])
                        #print("torch.eq(gt_label,label_pre[action_index_T]):",torch.eq(gt_label,label_pre[action_index_T]))
                        if d_score_T[action_index_T] > self.tt:
                            reward = 1
                        else:
                            reward = -1
                        #print("D_score:",d_score_T[action_index_T][1],"reward:",reward)        
                        next_state_T = torch.tensor(state_T)            
                        next_state_T[action_index_T] = torch.zeros(1,next_state_T.shape[1])
                        if it==(self.select_num-1):
                            done = 1
                        else:
                            done = 0
                        self.replay_buffer_T.push(state_T,action_index_T,reward,next_state_T,done,select_list_T)
                        self.iter_dqn_T = self.iter_dqn_T+1
                        state_T = next_state_T
                        #print("select_list_T:",select_list_T)
                        #if len(self.replay_buffer_T)>cfg.BATCH_SIZE_DQN:
                        #    lossQ = DQN.compute_td_loss(self.current_model_T,self.target_model_T,self.replay_buffer_T,cfg.BATCH_SIZE_DQN)          
                        #if np.mod(self.iter_dqn_T,cfg.replace_target_iter)==0:
                        #    DQN.update_target(self.current_model_T,self.target_model_T)
                    select_index_T = select_index_T + [batch_idx_T[i] for i in select_list_T]
                if len(self.replay_buffer_T)>cfg.BATCH_SIZE_DQN:
                    lossQ = DQN.compute_td_loss(self.current_model_T,self.target_model_T,self.replay_buffer_T,cfg.BATCH_SIZE_DQN)          
                if np.mod(self.iter_dqn_T,cfg.replace_target_iter)==0:
                    DQN.update_target(self.current_model_T,self.target_model_T)
                #d_instance = d_instance[select_index_T]
                if img==0:
                    d_instance_refine = d_instance_q[img][select_index_T]
                else:
                    d_instance_refine = torch.cat((d_instance_refine,d_instance_q[img][select_index_T]),0)

        if target:
            return d_instance_refine,lossQ

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        if self.S_agent:
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, d_instance_refine, lossQ#,diff
        else:
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, d_instance, lossQ

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
