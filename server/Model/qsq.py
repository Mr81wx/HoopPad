import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import Adam


class Net(torch.nn.Module):
    def __init__(self,dim_in, dim_h, dim_out):
        super(Net, self).__init__()
        self.dim_h = dim_h
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.layer_A = nn.Sequential(nn.Linear(self.dim_in,self.dim_h),nn.LeakyReLU(),
                                    nn.Linear(self.dim_h,self.dim_h*2),nn.LeakyReLU(),
                                    nn.Linear(self.dim_h*2,1))
        
    def forward(self, data):
        x = data
        x = self.layer_A(x)
        #x = torch.clamp(x, min=0, max=100) 
    
        return x


def Qsq_feature_(offense_position,defense_position,distance_o2ball): #[B,,]
    #print(offense_position.shape,defense_position.shape,distance_o2ball.shape)
    distance_2_rim = torch.norm(offense_position[:,:,:2] - torch.tensor([6.0,25.0]).unsqueeze(0).unsqueeze(0).to(offense_position.device),dim=-1) #[5]
    closetDefDist,_ =  torch.min(torch.norm(offense_position[:,:,:2].unsqueeze(2) - defense_position.unsqueeze(1), dim=-1),dim=-1)
    closetDefDist = closetDefDist * 0.8 #- (closetDefDist / 35) * 7
    catchAndShoot = distance_o2ball > 3.2 #True for catch&shoot
    rim_position = torch.tensor([6.0,25.0]).unsqueeze(0).unsqueeze(1).to(offense_position.device)
    baseline_vec = torch.tensor([0,-1]).unsqueeze(0).unsqueeze(0).to(offense_position.device)
    shotAngle = F.cosine_similarity((offense_position[:,:,:2] - rim_position),baseline_vec,dim=-1)
    shooterSpeed = offense_position[:,:,2]
    
    #shotclock = shotclock.repeat(1, 5)
    #print('clock',shotclock.shape)

    print(closetDefDist.shape,distance_2_rim.shape,catchAndShoot.shape,shotAngle.shape,shooterSpeed.shape)
    #'shotClock','closestDefDist','distance','catchAndShoot','shotAngle','shooterSpeed'
    input_tensor = torch.cat((closetDefDist.unsqueeze(2),distance_2_rim.unsqueeze(2), catchAndShoot.unsqueeze(2), shotAngle.unsqueeze(2),shooterSpeed.unsqueeze(2)), dim=2).float()
    print(input_tensor.shape)
    return input_tensor

def cal_Qsq_batch_(state_frame,predict_frame,team_ids_batch,Qsq_model):
    #state_frame [A,3]
    #predict_frame [A,2]
    #team_ids_batch [A,1]
    #shotclock [B,1]
    #input state_batch(offense_position), defense_position, Qsq model
    offense_index = torch.nonzero(team_ids_batch == 1).squeeze() #B*11,1,4
    ball_index = torch.nonzero(team_ids_batch == 0).squeeze()
    defense_index = torch.nonzero(team_ids_batch == 2).squeeze() #B

    offense_position = state_frame[offense_index].view(-1,5,3) #B*5,1,3 (x,y,v)
    ball_position = state_frame[ball_index].view(-1,1,3) #B,1,3
    defense_position = predict_frame[defense_index].view(-1,5,2) #B*5,1,2

    distance_o2ball = torch.norm(offense_position[:,:,:2] - ball_position[:,:,:2],dim=-1)

    input_tensor = Qsq_feature_(offense_position,defense_position,distance_o2ball)
    qsq_values = Qsq_model(input_tensor) #[B,6,1]
    #print(qsq_values[0,:,:])
    Qsq = qsq_values.squeeze(2) 
    #print(Qsq[0,:])

    #Qsq_sum = torch.sum(Qsq,dim=1)
    #Xgb_model(input_tensor)/distance_o2ball

    return(Qsq)

class Qsq(pl.LightningModule):
    def __init__(self, model, optimizer=Adam, lr=1e-3):
        super(Qsq, self).__init__()
        self.model = model
        self.lr = lr
        
        self.loss_1 = nn.SmoothL1Loss(reduction='mean')
        self.optimizer = optimizer
        
    def forward(self, X):
        return self.model(X)
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        return optimizer

    def training_step(self, batch, batch_idx):
        X, gt = batch
        pre = self(X)
        
        loss_1 = self.loss_1(pre, gt.unsqueeze(1))
        self.log('train_loss_1', loss_1)
        
        return loss_1
        
    def validation_step(self, batch, batch_idx):
        X, gt = batch
        pre = self(X)
        
        loss_1 = self.loss_1(pre, gt.unsqueeze(1))
        self.log('val_loss_1', loss_1)
        
        return loss_1
    