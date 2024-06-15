const useCombineData = (response, type = 'init') => { 

    console.log(response.data);

    const combinedData = response.data.agent_IDs.map((agentId, index) => ({
        agent_id: agentId,
        real_T: response.data.real_T.data[index],
        ghost_T: response.data.ghost_T.data[index],
        teamID: response.data.player_detail[index][2],
        player_index: index,
        type : type,
        real_qsq:  index>0? response.data.real_QSQ[index-1]:"",
        ghost_qsq: index>0? response.data.ghost_QSQ[index-1]:"",

      }));
      return combinedData
 }


 export default useCombineData