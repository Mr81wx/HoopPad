const useCombineData = (response) => { 

    const combinedData = response.data.agent_IDs.map((agentId, index) => ({
        agent_id: agentId,
        real_T: response.data.real_T.data[index],
        ghost_T: response.data.ghost_T.data[index],
        teamID: response.data.player_detail[index][2],
        player_index: index 
      }));
      return combinedData
 }


 export default useCombineData