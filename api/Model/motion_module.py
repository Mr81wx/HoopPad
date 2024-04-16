import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.functional as FU
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import sys, os
import os.path as osp
import numpy as np
#import cv2
import copy
#import hydra
import pytorch_lightning as pl
from .utils import *




class Scene_Motion(pl.LightningModule):
    def __init__(self, model,optimizer=Adam,lr=1e-3):
        super(Scene_Motion, self).__init__()
        self.optimizer = optimizer
        self.model = model.to(self.device)
        self.lr = lr
        self.F = 6
        self.target_scale = 1.0
        
        
    def forward(self, states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, agent_ids_batch,team_ids_batch):

        out = self.model(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, agent_ids_batch,team_ids_batch)
        

        #decoding = self.decoder(encodings, agents_batch_mask, states_padding_mask_batch)
        
        return out  #输出prediction
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9,0.999))
        #scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

        return [optimizer]

    def training_step(self, batch, batch_idx):

        states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, num_agents_accum, agent_ids_batch,team_ids_batch,labels_batch = batch
        # get classfication
        out = self(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, agent_ids_batch,team_ids_batch)
        #print(out,labels_batch)
        
        prediction = out
        to_predict_mask = ~states_padding_mask_batch*states_hidden_mask_batch
        gt = states_batch[:,:,:2] #A,T,2
        gt = gt.unsqueeze(2).repeat(1,1,self.F,1) #[A,T,2] -> [A,T,F,2]
        gt_scale = torch.ones_like(gt).fill_(self.target_scale)
        gt_dist = Laplace(gt, gt_scale)# [A,T,F,2]
        
        
        predict_dist = Laplace(prediction[:, :, :,0::2], prediction[:,:,:,1::2])  # [A,T,F,2]
        loss = torch.distributions.kl.kl_divergence(gt_dist, predict_dist) #[A,T,F,2]
            
        loss_mask = to_predict_mask.unsqueeze(2).unsqueeze(3).repeat(1, 1,self.F, 2) #[A,T] -> [A,T,F,2]
            
        loss_ = loss * loss_mask #[A,T,F,2]
        loss_ = torch.sum(loss_, dim=3) #[A,T,F]
        loss_ = torch.mean(loss_, dim=1) #mean each trajectory [A,F]
           

            # per agent (so min across the future dimension). [A,F]
        marginal_loss = torch.min(loss_,dim = 1).values#[A]
        marginal_loss_ = torch.sum(marginal_loss) #value
            
            #P_loss = loss_.view(-1, 6 , self.F) #[B,A,F]
            #joint_loss = torch.sum(P_loss,dim=1) #[B,F]
        joint_loss = torch.sum(loss_,dim = 0) #[F]
        joint_loss_ = torch.min(joint_loss)#
            #joint_loss_ = torch.mean(joint_loss_)

        MR_loss = marginal_loss_ + 0.1*joint_loss_
            
        summary_loss = MR_loss
        
        self.log_dict({'train/MR_loss':summary_loss})
        
        self.logger.experiment.add_scalar('Loss/train', summary_loss.item(), self.global_step)
        
        print('summary_loss:', summary_loss)
        
        # return {'batch': batch, 'pred': prediction, 'gt': gt, 'loss': summary_loss 'att_weights': out['att_weights']}
        return summary_loss

        


    def validation_step(self, batch, batch_idx):
        print(len(batch))
        states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, num_agents_accum, agent_ids_batch, team_ids_batch,labels_batch = batch
        
        # get classfication
        out = self(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch,agent_ids_batch,team_ids_batch)

        prediction = out
        to_predict_mask = ~states_padding_mask_batch*states_hidden_mask_batch
        gt = states_batch[:,:,:2] #A,T,2
        gt = gt.unsqueeze(2).repeat(1,1,self.F,1) #[A,T,2] -> [A,T,F,2]
        
        rs_error = ((prediction[:,:,:,0::2] - gt) ** 2).sum(dim=-1).sqrt_() 
        rs_error[~to_predict_mask]=0 #[A,T,F]
        
        rse_sum_1 = torch.sum(rs_error[:,11:16,:],dim=1) #[A,F]
        rse_sum_2 = torch.sum(rs_error[:,11:21,:],dim=1)
        rse_sum_3 = torch.sum(rs_error[:,11:26,:],dim=1)
        rse_sum_4 = torch.sum(rs_error[:,11:31,:],dim=1)
        
        
        #1s,2s,3s,4s
        ade_mask = to_predict_mask.sum(-1)!=0 #[A]
        print(rse_sum_1[ade_mask].shape)
        ade_1 = (rse_sum_1[ade_mask].permute(1,0)/to_predict_mask[ade_mask].sum(-1)).permute(1,0)
        ade_2 = (rse_sum_2[ade_mask].permute(1,0)/to_predict_mask[ade_mask].sum(-1)).permute(1,0)
        ade_3 = (rse_sum_3[ade_mask].permute(1,0)/to_predict_mask[ade_mask].sum(-1)).permute(1,0)
        ade_4 = (rse_sum_4[ade_mask].permute(1,0)/to_predict_mask[ade_mask].sum(-1)).permute(1,0)
        
        
        fde_mask = to_predict_mask.sum(-1)!=0
        fde_1 = rs_error[fde_mask][:,15,:]
        fde_2 = rs_error[fde_mask][:,20,:]
        fde_3 = rs_error[fde_mask][:,25,:]
        fde_4 = rs_error[fde_mask][:,30,:]

        minade_1, _ = ade_1.min(dim=-1)
        minade_2, _ = ade_2.min(dim=-1)
        minade_3, _ = ade_3.min(dim=-1)
        minade_4, _ = ade_4.min(dim=-1)
        
        minfde_1, _ = fde_1.min(dim=-1)
        minfde_2, _ = fde_2.min(dim=-1)
        minfde_3, _ = fde_3.min(dim=-1)
        minfde_4, _ = fde_4.min(dim=-1)
        
        batch_minade_1 = minade_1.mean()
        batch_minade_2 = minade_2.mean()
        batch_minade_3 = minade_3.mean()
        batch_minade_4 = minade_4.mean()
        
        batch_minfde_1 = minfde_1.mean()
        batch_minfde_2 = minfde_2.mean()
        batch_minfde_3 = minfde_3.mean()
        batch_minfde_4 = minfde_4.mean()
        
    
        # 记录准确率和 Top-3 准确率
        self.log_dict({'minade_1': batch_minade_1, 'minade_2': batch_minade_2,'minade_3': batch_minade_3,'minade_4': batch_minade_4,
            'minfde_1': batch_minfde_1, 'minfde_2': batch_minfde_2, 'minfde_3': batch_minfde_3, 'minfde_4': batch_minfde_4
        })
        
        
        # 将准确率添加到日志中
        self.logger.experiment.add_scalar('minade_1', batch_minade_1, self.global_step)
        self.logger.experiment.add_scalar('minade_2', batch_minade_2, self.global_step)
        self.logger.experiment.add_scalar('minade_3', batch_minade_3, self.global_step)
        self.logger.experiment.add_scalar('minade_4', batch_minade_4, self.global_step)
        self.logger.experiment.add_scalar('minfde_1', batch_minfde_1, self.global_step)
        self.logger.experiment.add_scalar('minfde_2', batch_minfde_2, self.global_step)
        self.logger.experiment.add_scalar('minfde_3', batch_minfde_3, self.global_step)
        self.logger.experiment.add_scalar('minfde_4', batch_minade_4, self.global_step)
        
        return {'minade_1': batch_minade_1, 'minade_2': batch_minade_2,'minade_3': batch_minade_3,'minade_4': batch_minade_4,
            'minfde_1': batch_minfde_1, 'minfde_2': batch_minfde_2, 'minfde_3': batch_minfde_3, 'minfde_4': batch_minfde_4
        }


class test_model_motion(nn.Module):
    def __init__(self, base_model, freeze=True, num_unfreeze = 0): #out_dim = number of labels
        super(test_model_motion, self).__init__()    
        
        self.encoder = base_model.encoder 
        self.decoder = base_model.decoder_MR
        for param in self.decoder.parameters():
            param.requires_grad_(True)
            #nn.init.normal_(param)
        # 最后一个线性层的输入特征数
        num_ftrs = base_model.decoder_CL.layer_MLP[-1].out_features
        # 移植Scene_transformer的encoder
        
        if freeze:
            self._freeze(num_unfreeze)
        

    def _freeze(self,n):
        num_layers = len(list(self.encoder.children())) # 14 层，冻结除最后 n+2 层之外的所有层
        current_layer = 1
        for child in list(self.encoder.children()):
            if current_layer > num_layers-(n): #数字代表不冻结的层数
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False
            current_layer += 1
    
    def forward(self, state_feat, agent_mask, padding_mask, hidden_mask,agent_ids_batch,team_ids_batch):
        output = self.encoder(state_feat, agent_mask, padding_mask, hidden_mask,agent_ids_batch)
        h = output['out']
        decoding = self.decoder(h, agent_mask, padding_mask,hidden_mask)
        
       
    
        return decoding.permute(1,2,0,3)
        
