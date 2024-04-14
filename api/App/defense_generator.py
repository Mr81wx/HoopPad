
import numpy as np
import torch
import pickle
from ..Data.dataload import create_def
from ..Model.module_all import HoopTransformer
from ..Model.motion_module import *
from ..Model.decoder import *
from ..Model.possession import *

def load_possession(possession_path):
    with open(possession_path, 'rb') as f:
        possession = pickle.load(f)
    states_batch, agents_batch_mask, states_padding_batch, states_hidden_batch,num_agents_accum,agent_ids_batch,team_ids_batch,labels_batch = create_def([possession])

    return states_batch, agents_batch_mask, states_padding_batch, states_hidden_batch,num_agents_accum,agent_ids_batch,team_ids_batch,labels_batch


def load_model(ckp_path):
    #load Model
    Scene_model = HoopTransformer(3,121,256,4,4,6,50,0.1,32,5e-5,[1])
    base_model = test_model_motion(Scene_model,freeze=True, num_unfreeze = 0) #Encoder freeze
    Test_model = Scene_Motion(model=base_model,lr=1e-3)
    checkpoint = torch.load(ckp_path, map_location=lambda storage, loc: storage)
    Test_model.load_state_dict(checkpoint['state_dict'])

    return Test_model

# input real off_players + ball trajectories and particial def_players trajectory (first 1s)
def def_gen(model,
states_batch, agents_batch_mask, states_padding_batch, states_hidden_batch,agent_ids_batch,team_ids_batch):
    
    model.to('cpu')
    model.eval()

    #get_local.clear() for attention map

    with torch.no_grad():
        out = model(states_batch, agents_batch_mask, states_padding_batch, states_hidden_batch,agent_ids_batch,team_ids_batch)
        # out [A,T,4] 第0个和第2个是x,y
    ghost_trajectory = out[:,:,0,::2] #[A,T,F]

    return states_batch, ghost_trajectory


if __name__ == "__main__":
    possession_path = 'api/Data/object_0.pkl'
    states_batch, agents_batch_mask, states_padding_batch, states_hidden_batch,agent_ids_batch,team_ids_batch = load_possession(possession_path)
    
    ckp_path = 'api/Checkpoints/V3_prompt+def_loss.ckpt'
    model = load_model(ckp_path)
    
   
    real_T,ghost_T = def_gen(model,states_batch, agents_batch_mask, states_padding_batch, states_hidden_batch,agent_ids_batch,team_ids_batch)