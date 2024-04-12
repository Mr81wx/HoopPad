import torch
import torch.nn as nn
import torch.nn.functional as FU

from Model.utils import *

from collections import OrderedDict
import copy
import numpy as np

class Decoder_MR(nn.Module):
    def __init__(self, device, time_steps=121, feature_dim=256, head_num=4, k=4, F=6):
        super().__init__()
        self.device = device
        self.time_steps = time_steps                # T
        self.feature_dim = feature_dim              # D
        self.head_num = head_num                    # H
        self.k = k
        self.F = F

        #self.layer_R = SelfAttLayer_Enc(self.time_steps,self.feature_dim,self.head_num,self.k,across_time=True) 
        #self.layer_S = SelfAttLayer_Enc(self.time_steps,self.feature_dim,self.head_num,self.k,across_time=False)
       
        onehots_ = torch.tensor(range(F))
        self.onehots_ = FU.one_hot(onehots_, num_classes=F).to(self.device)

        self.layer_T = nn.Sequential(nn.Linear(self.feature_dim+self.F,feature_dim), nn.ReLU())
        #self.layer_T.apply(init_xavier_glorot)

        self.layer_U = SelfAttLayer_Dec(self.time_steps,self.feature_dim,self.head_num,self.k,across_time=True)
        self.layer_V = SelfAttLayer_Dec(self.time_steps,self.feature_dim,self.head_num,self.k,across_time=False)

        self.layer_W = SelfAttLayer_Dec(self.time_steps,self.feature_dim,self.head_num,self.k,across_time=True)
        self.layer_X = SelfAttLayer_Dec(self.time_steps,self.feature_dim,self.head_num,self.k,across_time=False)

        self.layer_Y = nn.LayerNorm(self.feature_dim)

        self.layer_Z1 = nn.Sequential(nn.Linear(self.feature_dim,4), nn.ReLU(), Permute4Batchnorm((1,3,0,2)),
                            nn.BatchNorm2d(4),nn.Softplus(), Permute4Batchnorm((2,0,3,1)))  #最后多一个softplus是为了保证输出参数都是>0 
                            #输出为x,y的laplace分布参数 4个 x.loc,x.scale,y.loc,y.scale
        self.layer_Z1.apply(init_xavier_glorot)
        #self.layer_Z2 = nn.Linear(4 ,2)  # 输出应该是x, y
    def forward(self, state_feat, batch_mask, padding_mask, hidden_mask=None):
        A,T,D = state_feat.shape
        assert (T==self.time_steps and D==self.feature_dim)
        #state_feat = state_feat.reshape((A,T,-1,self.F))
        #x = state_feat.permute(3,0,1,2)

        '''onehots_ = copy.deepcopy(self.onehots_)
        onehots_ = onehots_.repeat(1,A,T,1)
        onehots_ = onehots_.to(state_feat.device)
        # x = state_feat.unsqueeze(0).repeat(self.F,1,1,1)    # [F,A,T,D]

        x = torch.cat((x,onehots_),dim=-1)                  # [F,A,T,D+F]
        x = self.layer_T(x)                                 # [F,A,T,D]
        '''
        #output = self.layer_R(state_feat,batch_mask, padding_mask, hidden_mask)
        #output = self.layer_S(output,batch_mask, padding_mask, hidden_mask)

        onehots_ = self.onehots_.view(self.F,1,1,self.F).repeat(1,A,T,1)
        onehots_ = onehots_.to(state_feat.device)
        x = state_feat.unsqueeze(0).repeat(self.F,1,1,1)

        x = torch.cat((x,onehots_),dim=-1)
        x = self.layer_T(x)

        x = self.layer_U(x,batch_mask=batch_mask)#, padding_mask=padding_mask)
        x = self.layer_V(x,batch_mask=batch_mask)#, padding_mask=padding_mask)
        
        x = self.layer_W(x,batch_mask=batch_mask)#, padding_mask=padding_mask)
        x = self.layer_X(x,batch_mask=batch_mask)#, padding_mask=padding_mask)

        x = self.layer_Y(x)
        x = self.layer_Z1(x)
        #x = self.layer_Z2(x)                                # [F,A,T,D]
        
        return x



class Decoder_CL(nn.Module): #Contrastive learning
    def __init__(self, device,time_steps=121, feature_dim=256, out_dim = 128):
        super().__init__()
        self.time_steps = time_steps                # T
        self.feature_dim = feature_dim              # D
        self.layer_MLP = nn.Sequential(nn.Linear(feature_dim,feature_dim*2),
                            nn.BatchNorm1d(2*feature_dim), nn.SELU(),
                            nn.Linear(feature_dim*2,feature_dim),
                            nn.BatchNorm1d(feature_dim),nn.SELU(),
                            nn.Linear(feature_dim,out_dim))
        
        
    def forward(self, state_feat, team_ids):
        #state_feat [batch_size*11*2,121,256] team_ids [batch_size*11*2]
        
        # # team_ids = (team_ids == 0) | (team_ids == 1) # filter ball and off
        # team_ids = (team_ids != 2)
        # h = state_feat[team_ids,...] #[batch_size*6*2,121,256] filter defenders
        # h = h.reshape(-1,6,self.time_steps,self.feature_dim) # shape [Batch_size*2,11,121,256]
        # h = h.permute(0, 3, 2, 1) #[batch_size*2,256,121,6]
        # h = FU.avg_pool2d(h, (self.time_steps,6)) #pooling -> [batch_size*2,256]
        team_ids = (team_ids == 0) | (team_ids == 1) #filter ball and off
        team_ids = team_ids.unsqueeze(-1).unsqueeze(-1).expand(-1,self.time_steps,self.feature_dim) #[batch_size*11*2,121,256]

        h = state_feat*team_ids #[batch_size*6*2,121,256] filter defenders
        h = h.reshape(-1,11,self.time_steps,self.feature_dim) # shape [Batch_size*2,11,121,256]
        h = h.permute(0, 3, 2, 1) #[batch_size*2,256,121,11]
        h = FU.avg_pool2d(h, (self.time_steps,11)) #pooling -> [batch_size*2,256]

        # split h according  0 dimen
        split_index = h.size(0) // 2
        out_x = h[:split_index] #[Batch_size,256]
        out_y = h[split_index:] #[Batch_size,256]
       
        
        zis = FU.normalize(out_x, dim=1) # [20, 256, 1, 1]
        zis = torch.squeeze(zis) # [20, 256]
        zjs = FU.normalize(out_y, dim=1)
        zjs = torch.squeeze(zjs)

        x = self.layer_MLP(zis) # [Batch_size,128]
        y = self.layer_MLP(zjs) # [Batch_size,128]
        
        return x,y



class Decoder_DT(nn.Module): #Discriminating Teams
    def __init__(self, device,out_dim = 3):
        super().__init__()
        self.out_dim = out_dim
        self.classifier = nn.Sequential(
                                        nn.Linear(256, 128),  # 输入大小为 256，隐藏层大小为 128
                                        nn.ReLU(),            # 使用 ReLU 作为激活函数
                                        nn.Linear(128, self.out_dim)    # output 3 classes
                                    )
        
    def forward(self, state_feat):
        #state_feat [batch_size*11*2,121,256] team_ids [batch_size*11]
        x = torch.mean(state_feat,dim=[1]) #[batch_size*11*2,256]
        x = self.classifier(x) #[batch_size*11*2,3]
        
        return x
        
        
    
        
