
import sys
sys.path.append("..") 
import numpy as np
import torch
import pickle
from Data.dataload import create_def
from Model.module_all import HoopTransformer
from Model.motion_module import *
from Model.decoder import *
from Model.possession import *

def load_possession(possession_path):
    with open(possession_path, 'rb') as f:
        possession = pickle.load(f)
    states_batch, agents_batch_mask, states_padding_batch, states_hidden_batch,num_agents_accum,agent_ids_batch,team_ids_batch,team_name_batch,labels_batch = create_def([possession])

    return states_batch, agents_batch_mask, states_padding_batch, states_hidden_batch,num_agents_accum,agent_ids_batch,team_ids_batch,team_name_batch,labels_batch


def load_model(ckp_path):
    #load Model
    Scene_model = HoopTransformer(3,121,256,4,4,6,50,0.1,32,5e-5,[1])
    base_model = test_model_motion(Scene_model,freeze=True, num_unfreeze = 0) #Encoder freeze
    decoder_init = Decoder_MR(device = 'cpu',time_steps=121, feature_dim=256, head_num=4, k=6, F=1)
    base_model.decoder = decoder_init
    Test_model = Scene_Motion(model=base_model,lr=1e-3)
    checkpoint = torch.load(ckp_path, map_location=lambda storage, loc: storage)
    Test_model.load_state_dict(checkpoint['state_dict'],strict=False)

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
    ghost_trajectory = out[:,:,0,::2] #[A,T,F,3]
    print(ghost_trajectory.shape)
    ghost_trajectory = ghost_trajectory[:,:,:2] #[A,T,F,2]
    off_index = torch.where(team_ids_batch != 2)[0]
    ghost_trajectory[off_index] = states_batch[:,:,:2][off_index]

    return states_batch, ghost_trajectory





def get_results(possession_path, ckp_path):
    states_batch, agents_batch_mask, states_padding_batch, states_hidden_batch,num_agents_accum,agent_ids_batch,team_ids_batch,team_name_batch,labels_batch = load_possession(possession_path)
    model = load_model(ckp_path)
    real_T,ghost_T = def_gen(model,states_batch, agents_batch_mask, states_padding_batch, states_hidden_batch,agent_ids_batch,team_ids_batch)
    
    team_ids_batch = team_ids_batch.numpy().astype(int)
    agent_ids_batch = agent_ids_batch.numpy().astype(int)
    players_detail = np.stack((team_ids_batch.astype(int), agent_ids_batch.astype(int), team_name_batch.astype(int)), axis=1)

    print(players_detail)

    return {"real_T": real_T, "ghost_T": ghost_T, "team_IDs": team_ids_batch, "agent_IDs":agent_ids_batch, "player_detail": players_detail}


# def update_prompt(back_list,states_batch,states_hidden_batch): 
#     #[frame_number,player_number_new_x,new_y] need to be pass from front 
#     # update x,y
#     for sub_list in back_list:    
#         frame_number = sub_list[0]
#         player_number = sub_list[1]
#         new_x = sub_list[2]
#         new_y = sub_list[3]
#     states_batch[player_number,frame_number,0] = new_x
#     states_batch[player_number,frame_number,1] = new_y
#     #update hidden mask, make this position visible to model
#     states_hidden_batch[palyer_number,frame_number] = False

#     return states_batch, states_hidden_batch


def update_results(back_list,possession_path, ckp_path): 
    #[frame_number,player_number_new_x,new_y] need to be pass from front 
    # update x,y
    
    print(back_list)
    
    states_batch, agents_batch_mask, states_padding_batch, states_hidden_batch,num_agents_accum,agent_ids_batch,team_ids_batch,team_name_batch,labels_batch = load_possession(possession_path)
    model = load_model(ckp_path)
    
    team_ids_batch = team_ids_batch.numpy().astype(int)
    agent_ids_batch = agent_ids_batch.numpy().astype(int)
    players_detail = np.stack((team_ids_batch.astype(int), agent_ids_batch.astype(int), team_name_batch.astype(int)), axis=1)

    # print(players_detail)


    for sub_list in back_list:    
        frame_number = sub_list[0]
        player_number = sub_list[1]
        new_x = sub_list[2]
        new_y = sub_list[3]
    
    print("state_batch",states_batch)
    
    states_batch[player_number,frame_number,0] = new_x
    states_batch[player_number,frame_number,1] = new_y
    #update hidden mask, make this position visible to model
    states_hidden_batch[player_number,frame_number] = False
    
    real_T,ghost_T = def_gen(model,states_batch, agents_batch_mask, states_padding_batch, states_hidden_batch,agent_ids_batch,team_ids_batch)

    return {"real_T": real_T, "ghost_T": ghost_T, "team_IDs": team_ids_batch, "agent_IDs":agent_ids_batch, "player_detail": players_detail}


# if __name__ == "__main__":
#     #possession_path = '/Users/yufu/Documents/Code/HoopPad/server/Data/object_0.pkl'
#     possession_path = '/workspaces/HoopPad/server/Data/object_0.pkl'
#     states_batch, agents_batch_mask, states_padding_batch, states_hidden_batch,num_agents_accum,agent_ids_batch,team_ids_batch,team_name_batch,labels_batch = load_possession(possession_path)
    
#     #ckp_path = '/Users/yufu/Documents/Code/HoopPad/server/Checkpoints/V3_prompt+def_loss.ckpt'
#     ckp_path = '/workspaces/HoopPad/server/Checkpoints/BC.ckpt'
#     model = load_model(ckp_path)
#     real_T,ghost_T = def_gen(model,states_batch, agents_batch_mask, states_padding_batch, states_hidden_batch,agent_ids_batch,team_ids_batch)
    
#     team_ids_batch = team_ids_batch.numpy()
#     agent_ids_batch = agent_ids_batch.numpy()
#     players_detail = np.stack((team_ids_batch.astype(int), agent_ids_batch.astype(int), team_name_batch.astype(int)), axis=1)
#     #print(agent_ids_batch)
#     np.set_printoptions(suppress=True)
#     print(players_detail)
#     #print(ghost_T)