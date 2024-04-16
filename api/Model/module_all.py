import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.functional as FU
from pytorch_lightning.loggers import TensorBoardLogger
from ..Model.utils import laplace_kl, NTXentLoss
from torch.distributions.laplace import Laplace
from torch.optim.lr_scheduler import OneCycleLR

import sys, os
import os.path as osp
import numpy as np
#import cv2
import copy
#import hydra
import pytorch_lightning as pl


from ..Model.encoder import Encoder
from ..Model.decoder import Decoder_MR, Decoder_CL, Decoder_DT
#from datautil.waymo_dataset import xy_to_pixel

#COLORS = [(0,0,255), (255,0,255), (180,180,0), (143,143,188), (0,100,0), (128,128,0)]
#TrajCOLORS = [(0,0,255), (200,0,0), (200,200,0), (0,200,0)]

class HoopTransformer(pl.LightningModule):
    def __init__(self, in_feat_dim,time_steps,feature_dim,head_num,k,F,halfwidth,temperature,batchsize,lr,tasks):
        super(HoopTransformer, self).__init__()
        #self.cfg = cfg
        self.in_feat_dim = in_feat_dim
        self.time_steps = time_steps
        #self.current_step = cfg.model.current_step
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.k = k
        self.F = F
        self.halfwidth = halfwidth 
        self.target_scale = 1.0 #Laplace Target Scale
        self.temperature = temperature #nextloss
        self.batchsize = batchsize #nextloss
        self.lr = lr
        
        self.tasks = tasks #[1,2,3]

        self.Loss = nn.MSELoss(reduction='none')

        self.encoder = Encoder(self.device, self.in_feat_dim, 
                                    self.time_steps, self.feature_dim, self.head_num)
        self.decoder_MR = Decoder_MR(self.device, self.time_steps, self.feature_dim, self.head_num, self.k, self.F)
        self.decoder_CL = Decoder_CL(self.device,self.time_steps,self.feature_dim)
        self.decoder_DT = Decoder_DT(self.device,3)
        
        self.NTXentLoss = NTXentLoss(device='cuda',batch_size=self.batchsize,temperature=self.temperature)
        
        self.weight = torch.tensor([11,11/5,11/5],dtype=torch.float)
        self.DT_loss = nn.CrossEntropyLoss()
        
        ### viz options
        #self.width = cfg.viz.width
        #self.totensor = transforms.ToTensor()
        
    def forward(self, states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, agent_ids_batch,team_ids_batch): 
        
        e = self.encoder(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, agent_ids_batch)
        encodings = e['out']
        encodings_CL = e['out_layer_O']
        # states_padding_mask_batch = states_padding_mask_batch + ~states_hidden_mask_batch
        if 1 in self.tasks:
            decoding_1 = self.decoder_MR(encodings, agents_batch_mask, states_padding_mask_batch,states_hidden_mask_batch)
        else:
            decoding_1 = torch.zeros(1,1,1,1)
        
        if 2 in self.tasks:
            decoding_2 = self.decoder_CL(encodings_CL,team_ids_batch)
        else:
            decoding_2 = 0
            
        if 3 in self.tasks:
            decoding_3 = self.decoder_DT(encodings)
        else:
            decoding_3 = 0
        
        return {'MR_result': decoding_1.permute(1,2,0,3), 'att_weights': e['att_weights'],'CL_result': decoding_2, 'DT_result': decoding_3} #[F,A,T,4]转换为[A,T,F,4]

    def training_step(self, batch, batch_idx):

        '''states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, \
                    roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch, \
                        agent_rg_mask, agent_traffic_mask, (num_agents_accum, num_rg_accum, num_tl_accum), \
                            sdc_masks, center_ps = batch.values()'''
                            
        states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, num_agents_accum, agent_ids_batch, team_ids_batch = batch
        
        #print(torch.mean(states_hidden_mask_batch.float()).item())
        '''
        no_nonpad_mask = torch.sum((states_padding_mask_batch*~states_hidden_mask_batch),dim=-1) != 0
        states_batch = states_batch[no_nonpad_mask]
        agents_batch_mask = agents_batch_mask[no_nonpad_mask][:,no_nonpad_mask]
        states_padding_mask_batch = states_padding_mask_batch[no_nonpad_mask]
        states_hidden_mask_batch = states_hidden_mask_batch[no_nonpad_mask]
        '''
        #agent_ids_batch = agent_ids_batch[no_nonpad_mask]
        
        
        out = self(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, agent_ids_batch, team_ids_batch)
        # MR task
        if 1 in self.tasks:
            
            # Calculate Loss
            prediction = out['MR_result'] #[A,T,F,4]
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
        
        else:
            MR_loss = 0
            
        
        if 2 in self.tasks:
        #CL task
            zis, zjs = out['CL_result']
            CL_loss = self.NTXentLoss(zis,zjs)
        else:
            CL_loss = 0
        
        if 3 in self.tasks:
            team_ids_batch_index = (team_ids_batch != -1) 
            team_ids_batch = team_ids_batch[team_ids_batch_index]
            #DT task
            predict_team = out['DT_result']
            predict_team = predict_team[team_ids_batch_index]
            DT_loss = self.DT_loss(predict_team,team_ids_batch.to(torch.int64))
        else:
            DT_loss = 0
        
        MR_weight, CL_weight, DT_weight = 1, 5e3, 5e3
        summary_loss = MR_loss*MR_weight + CL_loss*CL_weight + DT_loss*DT_weight # TODO: weight 
        
        self.log_dict({'train/MR_loss':MR_loss,'train/CL_loss':CL_loss,'train/DT_loss':DT_loss})
        
        self.logger.experiment.add_scalar('Loss/train', summary_loss.item(), self.global_step)
        
        print('summary_loss:', summary_loss, 'MR:', MR_loss ,'CL:', CL_loss,'DT:', DT_loss)
        
        # return {'batch': batch, 'pred': prediction, 'gt': gt, 'loss': summary_loss 'att_weights': out['att_weights']}
        return summary_loss

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            print(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

    def validation_step(self, batch, batch_idx):
        
        states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, num_agents_accum, agent_ids_batch, team_ids_batch = batch
        
        
        '''
        no_nonpad_mask = torch.sum((states_padding_mask_batch*~states_hidden_mask_batch),dim=-1) != 0
        states_batch = states_batch[no_nonpad_mask]
        agents_batch_mask = agents_batch_mask[no_nonpad_mask][:,no_nonpad_mask]
        states_padding_mask_batch = states_padding_mask_batch[no_nonpad_mask]
        states_hidden_mask_batch = states_hidden_mask_batch[no_nonpad_mask]
        '''
        # Predict
        out = self(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, agent_ids_batch, team_ids_batch)

                # MR task
        if 1 in self.tasks:
            
            # Calculate Loss
            prediction = out['MR_result'] #[A,T,F,4]
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
        
            rs_error = ((prediction[:,:,:,0::2] - gt) ** 2).sum(dim=-1).sqrt_() #[A,T,F]
            rs_error[~to_predict_mask]=0 #[A,T,F]
            rse_sum = torch.sum(rs_error,dim=1) #[A,F]
            ade_mask = to_predict_mask.sum(-1)!=0 #[A]
            ade = (rse_sum[ade_mask].permute(1,0)/to_predict_mask[ade_mask].sum(-1)).permute(1,0)


            fde_mask = to_predict_mask[:,-1]==True
            fde = rs_error[fde_mask][:,-1,:]

            minade, _ = ade.min(dim=-1)
            avgade = ade.mean(dim=-1)
            minfde, _ = fde.min(dim=-1)
            avgfde = fde.mean(dim=-1)

            batch_minade = minade.mean()
            batch_minfde = minfde.mean()
            batch_avgade = avgade.mean()
            batch_avgfde = avgfde.mean()

        else:
            MR_loss = 0
            batch_minade = 0
            batch_minfde = 0
            batch_avgade = 0
            batch_avgfde = 0 
        
        if 2 in self.tasks:
        #CL task
            zis, zjs = out['CL_result']
            CL_loss = self.NTXentLoss(zis,zjs)
        else:
            CL_loss = 0
        
        if 3 in self.tasks:
            team_ids_batch_index = (team_ids_batch != -1) 
            team_ids_batch = team_ids_batch[team_ids_batch_index]
            #DT task
            predict_team = out['DT_result']
            predict_team = predict_team[team_ids_batch_index]
            DT_loss = self.DT_loss(predict_team,team_ids_batch.to(torch.int64))
        else:
            DT_loss = 0
        
        MR_weight, CL_weight, DT_weight = 1, 5e3, 5e3
        summary_loss = MR_loss*MR_weight + CL_loss*CL_weight + DT_loss*DT_weight # TODO: weight 

        #
        

        self.log_dict({'val/loss': MR_loss, 'val/minade': batch_minade, 'val/minfde': batch_minfde, 'val/avgade': batch_avgade, 'val/avgfde': batch_avgfde,'val/DT_loss': DT_loss,'val/CL_loss':CL_loss})

        self.val_out =  {'states': states_batch, 'states_padding': states_padding_mask_batch, 'states_hidden': states_hidden_mask_batch, 
                        'num_agents_accum': num_agents_accum,
                        'loss': summary_loss, 'att_weights': out['att_weights']}
        self.logger.experiment.add_scalar('Loss/valid', summary_loss, self.global_step)
        return summary_loss



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9,0.999)) #1e-7
        #scheduler = OneCycleLR(optimizer, max_lr = 5e-7,steps_per_epoch=8000, epochs=50)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        lr = scheduler.get_last_lr()[0]
        return [optimizer], [scheduler]
        #return optimizer

'''@hydra.main(config_path='../conf', config_name='config.yaml')
def test_valend(cfg):
    from datautil.waymo_dataset import WaymoDataset, waymo_collate_fn, waymo_worker_fn
    from model.pl_module import SceneTransformer
    torch.multiprocessing.set_sharing_strategy('file_system')
    pl.seed_everything(1235)
    # trainer_args = {}
    trainer_args = {'max_epochs': cfg.max_epochs,
                    'gpus': [0],#cfg.gpu_ids,
                    'accelerator': 'ddp',
                    'val_check_interval': 0.2, 'limit_train_batches': 1.0, 
                    'limit_val_batches': 0.001,
                    'log_every_n_steps': 100, 'auto_lr_find': True,
                    }

    trainer = pl.Trainer(**trainer_args)
    model = SceneTransformer(cfg)
    # model = model.load_from_checkpoint(checkpoint_path=cfg.dataset.test.ckpt_path, cfg=cfg)

    pwd = hydra.utils.get_original_cwd()
    dataset_valid = WaymoDataset(osp.join(pwd, cfg.dataset.valid.tfrecords), osp.join(pwd, cfg.dataset.valid.idxs), shuffle_queue_size=None)
    dloader_valid = DataLoader(dataset_valid, batch_size=cfg.dataset.valid.batchsize, shuffle=False, collate_fn=waymo_collate_fn, num_workers=0)

    trainer.validate(model=model, val_dataloaders=dloader_valid, verbose=True)
'''
if __name__ == '__main__':
    #test_valend()
    print('wx')
