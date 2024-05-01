const colorDict = {
    1610612737: ['#E13A3E', 'ATL'],
    1610612738: ['#008348', 'BOS'],
    1610612751: ['#061922', 'BKN'],
    1610612766: ['#1D1160', 'CHA'],
    1610612741: ['#CE1141', 'CHI'],
    1610612739: ['#860038', 'CLE'],
    1610612742: ['#007DC5', 'DAL'],
    1610612743: ['#4D90CD', 'DEN'],
    1610612765: ['#006BB6', 'DET'],
    1610612744: ['#FDB927', 'GSW'],
    1610612745: ['#CE1141', 'HOU'],
    1610612754: ['#00275D', 'IND'],
    1610612746: ['#ED174C', 'LAC'],
    1610612747: ['#552582', 'LAL'],
    1610612763: ['#0F586C', 'MEM'],
    1610612748: ['#98002E', 'MIA'],
    1610612749: ['#00471B', 'MIL'],
    1610612750: ['#005083', 'MIN'],
    1610612740: ['#002B5C', 'NOP'],
    1610612752: ['#006BB6', 'NYK'],
    1610612760: ['#007DC3', 'OKC'],
    1610612753: ['#007DC5', 'ORL'],
    1610612755: ['#006BB6', 'PHI'],
    1610612756: ['#1D1160', 'PHX'],
    1610612757: ['#E03A3E', 'POR'],
    1610612758: ['#724C9F', 'SAC'],
    1610612759: ['#BAC3C9', 'SAS'],
    1610612761: ['#CE1141', 'TOR'],
    1610612762: ['#00471B', 'UTA'],
    1610612764: ['#002B5C', 'WAS'],
};

// Function to get team color and abbreviation
function getTeamColor(id) {
    const teamInfo = colorDict[id];
    if (!teamInfo) {
        return null;  // Return null or throw an error if the ID is not found
    }
    return {
        color: teamInfo[0],
        abbreviation: teamInfo[1]
    };
}

export default getTeamColor;
