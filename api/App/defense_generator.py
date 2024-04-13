


import numpy as np
import torch


def get_newid(playerid,df):
    if playerid in df['playerid'].values:
        newid = df[df['playerid'] == playerid]['UnifiedPlayerID'].values[0]
    else:
        newid = df['UnifiedPlayerID'].max() + 1
        df.loc[len(df)] = {'playerid': playerid, 'UnifiedPlayerID': newid}

    return newid

def create_def(batch): #batch代表n个scene

    sample_freq = 5
    sequence_length = int(24*25/sample_freq + 1)
    list_24s = [i*(1/sample_freq) for i in range(0, sequence_length)]
    list_24s.reverse()

    time_steps = sequence_length
    states_batch = np.array([]).reshape(-1,time_steps,3)
    states_padding_batch = np.array([]).reshape(-1,time_steps)
    states_hidden_BP_batch = np.array([]).reshape(-1,time_steps)
    num_agents = np.array([])

    agent_ids_batch = np.array([]) #每一个batch的agent ids, = batch size * 11
    team_ids_batch = np.array([])

    player_newids = pd.read_csv('/content/drive/MyDrive/EPV/DataLoader/players.csv')
    for scene in batch:
        scene_tensor = None
        agent_ids = np.array([])
        team_ids = np.array([])
        for i in range(len(scene.agents)):
            agent = scene.agents[i]
            agent_id = get_newid(agent.playerid,player_newids)
            agent_ids = np.append(agent_ids,agent_id)
            #x = np.array(agent.x)[::sample_freq] #间隔sample_freq个取值
            #y = np.array(agent.y)[::sample_freq] #间隔sample_freq个取值
            #v = np.array(agent.v)[::sample_freq] #间隔sample_freq个取值
            #single_agent_tensor = torch.Tensor([[x,y,v]])

            agent_team = agent.teamid

            if agent_team == -1:
                team_ids= np.append(team_ids,0)
            elif agent_team == scene.off_teamid:
                team_ids = np.append(team_ids,1)
            else:
                team_ids = np.append(team_ids,2)

            single_agent_tensor = torch.Tensor([[x,y,v] for x,y,v in zip(agent.x,agent.y, agent.v)][::sample_freq])
            single_agent_tensor = single_agent_tensor.permute(1, 0)
            single_agent_tensor = torch.unsqueeze(single_agent_tensor, 0)
            single_agent_tensor = single_agent_tensor.to(torch.float)

            if scene_tensor == None:
                scene_tensor = single_agent_tensor
            else:
                scene_tensor = torch.cat([scene_tensor, single_agent_tensor], dim=0) #获得6个agent的tensor shape=[6,3,time_steps]
        #补齐A的维度为6
        A,D,T = scene_tensor.size()

        #补齐A的维度为11
        A,D,T = scene_tensor.size()
        # 如果agents 不足11个：用-1 补齐scence_tensor,agent_ids也用-1补齐
        if A < 11:
            new_dim = torch.full((11-A, D, T), -1)
            scene_tensor = torch.cat([new_dim, scene_tensor], dim=0)
            agent_ids = np.pad(agent_ids, (11 - A, 0), 'constant', constant_values=(-1,))
            team_ids = np.pad(team_ids, (11 - A, 0), 'constant', constant_values=(-1,))


        scene_tensor = torch.transpose(scene_tensor , dim0=1, dim1=2) #shape[A,T,D]

        time_24s = scene.time_24s[::sample_freq] #间隔sample_freq个取值
        if len(time_24s) < 30:
            continue

        end_padding_size =  121 - len(time_24s) #endtime～0

        states_feat = torch.nn.functional.pad(scene_tensor,  (0, 0, 1, end_padding_size-1, 0, 0), mode='constant', value=-1) #[6,121,3]
        states_padding = states_feat[:,:,0]
        states_padding = states_padding < 0 #bool型，[6,121] True为padding，ignore

        team_ids_batch = np.append(team_ids_batch,team_ids) # team_classification for task 3 (0-ball,1-off,2-def)
        def_index = np.where(team_ids_batch == 2)[0]

        states_hidden = np.zeros((11,121)).astype(np.bool_) # True 为hidden，这里不需要hidden,all false
        states_hidden[def_index,5:(T+1)] = True
        agent_ids_batch = np.append(agent_ids_batch,agent_ids) #player_embedding


        num_agents = np.append(num_agents, len(states_feat)) # numpy array(batch_size,) [11,11,11,11,11] #show which row is from same possession

        states_batch = np.concatenate((states_batch,states_feat), axis=0)
        states_padding_batch = np.concatenate((states_padding_batch,states_padding), axis=0)
        states_hidden_BP_batch = np.concatenate((states_hidden_BP_batch,states_hidden), axis=0)

    num_agents_accum = np.cumsum(np.insert(num_agents,0,0)).astype(np.int64)
    agents_batch_mask = np.ones((num_agents_accum[-1],num_agents_accum[-1])) #0代表一个回合内，需要attention

    for i in range(len(num_agents)):
        agents_batch_mask[num_agents_accum[i]:num_agents_accum[i+1], num_agents_accum[i]:num_agents_accum[i+1]] = 0

    states_batch = torch.FloatTensor(states_batch)
    agents_batch_mask = torch.BoolTensor(agents_batch_mask)
    states_padding_batch = torch.BoolTensor(states_padding_batch)
    states_hidden_BP_batch = torch.BoolTensor(states_hidden_BP_batch)

    agent_ids_batch = torch.FloatTensor(agent_ids_batch)
    team_ids_batch = torch.FloatTensor(team_ids_batch)
    labels_batch = torch.FloatTensor(team_ids_batch)


    return (states_batch, agents_batch_mask, states_padding_batch, states_hidden_BP_batch,num_agents_accum,agent_ids_batch,team_ids_batch,labels_batch)
