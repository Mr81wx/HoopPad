import { useRef, useEffect, useState } from "react";
import * as d3 from "d3";
import getTeamColor from "./userTeamColor";
import { BasketballCourt } from "./BasketballCourt";
import { Box, IconButton, Tooltip, Button, ButtonGroup } from "@mui/material";
import PlayCircleOutlineRoundedIcon from "@mui/icons-material/PlayCircleOutlineRounded";
import PauseCircleOutlineRoundedIcon from "@mui/icons-material/PauseCircleOutlineRounded";
import RestartAltRoundedIcon from "@mui/icons-material/RestartAltRounded";
import EventRepeatOutlinedIcon from '@mui/icons-material/EventRepeatOutlined';
import useCombineData from "../useCombineData";
import axios from "axios";

export const DrawPlayers = ({
  width,
  playerData,
  selectedDataFile,
  selectedCheckpoint,
}) => {
  const courtSVG = useRef();
  const [currentStep, setCurrentStep] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [T_type, setT_type] = useState("ghost_T");

  const [newPlayerData, setNewPlayerData] = useState([]);

  
  // const [backList, setBackList] = useState([]);

  let backList = [];
  const [newList, setNewList] = useState([]);

  useEffect(() => {
    if (playerData.length > 0) {
      setNewPlayerData(playerData);
    }
  }, [playerData]);

  useEffect(() => {
    console.log(newPlayerData);
    const courtWidth = width;
    const courtHeight = (courtWidth * 50) / 94;

    const svg = d3.select(courtSVG.current);
    svg.selectAll("*").remove();

    const courtItem = svg.append("g").attr("class", "courtGroup");

    const playerIMGWidth = 100;

    const groups = courtItem
      .selectAll("g")
      .data(playerData, d=> d.agent_id)
      .join("g")
      .attr(
        "transform",
        (d) =>
          `translate(${(d[T_type][1][0] * width) / 94}, ${
            (d[T_type][1][1] * width) / 94
          })`
      );

    const drag = d3
      .drag()
      .on("start", function (event, d) {
        d3.select(this).raise();
      })
      .on("drag", function (event, d) {
        d3.select(this).attr("transform", `translate(${event.x}, ${event.y})`);
      })
      .on("end", function (event, d) {
        console.log(d);
        const newX = (event.x * 94) / width;
        const newY = (event.y * 94) / width;

        const existingIndex = backList.findIndex(
          (entry) => entry[0] === currentStep && entry[1] === d.player_index
        );
        if (existingIndex >= 0) {
          backList[existingIndex] = [currentStep, d.player_index, newX, newY];
        } else {
          backList = [...backList, [currentStep, d.player_index, newX, newY]];
        }
        console.log(backList);

        setNewList(backList)
      });

    groups.call(drag);

    groups.each(function (d, i) {
      const group = d3.select(this);

      const radius = i === 0 ? 15 : 25;

      const clipPathId = `clip-path-${i}`;
      if (d.agent_id !== -1) {
        group
          .append("clipPath")
          .attr("id", clipPathId)
          .append("circle")
          .attr("r", radius)
          .attr("cx", 0)
          .attr("cy", 0);
      }

      console.log(getTeamColor(d.teamID)?.color);
      group
        .append("circle")
        .attr("r", radius)
        .style("fill", i === 0 ? "orange" : getTeamColor(d.teamID).color)
        .style("stroke-width", 3)
        .style("stroke", i === 0 ? "orange" : i > 5 ? "red" : "blue")
        .style("opacity", 0.8);

      const image = group
        .append("image")
        .attr("href", (d) =>
          d.agent_id === -1
            ? ""
            : `https://cdn.nba.com/headshots/nba/latest/1040x760/${d.agent_id}.png`
        )
        .attr("width", (d) => (d.agent_id === -1 ? 2 * radius : playerIMGWidth))
        .attr("height", (d) =>
          d.agent_id === -1 ? 2 * radius : playerIMGWidth
        )
        .attr("x", (d) =>
          d.agent_id === -1 ? -radius / 2 : -playerIMGWidth / 2
        )
        .attr("y", (d) =>
          d.agent_id === -1 ? -radius / 2 + 10 : -playerIMGWidth / 2 + 10
        )
        .style("opacity", 0.7);

      if (d.agent_id !== -1) {
        image.attr("clip-path", `url(#${clipPathId})`);
      }
    });
  }, [playerData, width, T_type]);

  const svg = d3.select(courtSVG.current);
  const groups = svg.select(".courtGroup").selectAll("g");

  // Handle the animation of players
  useEffect(() => {
    if (!isPlaying) return;

    console.log('newData')

    groups.data(newPlayerData, d => d.agent_id)


    async function movePlayersSequentially(startStep) {
      let index = startStep;
      while (index < 50 && isPlaying) {
        await new Promise((resolve) => {
          groups
            .transition()
            .duration(200)
            .ease(d3.easeLinear)
            .attr(
              "transform",
              (d) =>
                `translate(${(d[T_type][index][0] * width) / 94}, ${
                  (d[T_type][index][1] * width) / 94
                })`
            )
            .end()
            .then(resolve);
        });
        index++;

        setCurrentStep(index);

        if (currentStep === 49) {
          setIsPlaying(false);
          setCurrentStep(1);
        }
      }
    }

    movePlayersSequentially(currentStep);
  }, [isPlaying, currentStep, newPlayerData, width, T_type]);

  const handlePlayPause = () => {
    groups.interrupt();
    setIsPlaying(!isPlaying);
  };

  const handleRestart = () => {
    groups.interrupt();
    setIsPlaying(false);
    setCurrentStep(1);
    setIsPlaying(true);
  };

  console.log("backList", newList);

  const handleRerun = async () => {
    try {
      console.log("Back list:", newList);
      const response = await axios.post(
        `http://localhost:8080/api/update_data`,
        {
          backList: newList,
        },
        {
          params: {
            dataFile: selectedDataFile,
            checkpointFile: selectedCheckpoint,
          },
        }
      );

      console.log("Server response:", response.data);

      if (response.data && response.data.error) {
        console.error("Error from server:", response.data.error);
        return; 
      }

      const combinedData = useCombineData(response);
      console.log("Combined data:", combinedData);
      setNewPlayerData(combinedData); // Ensure setNewPlayerData is defined and used correctly
    } catch (error) {
      console.error("Error submitting selections:", error);
      if (error.response) {
        console.error("Error details:", error.response.data);
      }
    }
};


  return (
    <Box>
      <Box sx={{ mt: 2, display: "flex", justifyContent: "center", gap: 1 }}>
        {newList.length > 0 && (
          <Tooltip title="Re-calculate Ghost_T">
            <IconButton onClick={handleRerun} color="primary">       
                <EventRepeatOutlinedIcon />            
            </IconButton>
          </Tooltip>
        )}

        <Tooltip title="Play/Pause">
          <IconButton onClick={handlePlayPause} color="primary">
            {isPlaying ? (
              <PauseCircleOutlineRoundedIcon />
            ) : (
              <PlayCircleOutlineRoundedIcon />
            )}
          </IconButton>
        </Tooltip>
        <Tooltip title="Restart">
          <IconButton onClick={handleRestart} color="primary">
            <RestartAltRoundedIcon />
          </IconButton>
        </Tooltip>
        <ButtonGroup
          variant="text"
          color="primary"
          aria-label="trajectory type"
        >
          <Tooltip title="Real Trajectory">
            <Button
              onClick={() => setT_type("real_T")}
              variant={T_type === "real_T" ? "contained" : "outlined"}
            >
              Real T
            </Button>
          </Tooltip>
          <Tooltip title="Generated Trajectory">
            <Button
              onClick={() => setT_type("ghost_T")}
              variant={T_type === "ghost_T" ? "contained" : "outlined"}
            >
              Ghost T
            </Button>
          </Tooltip>
        </ButtonGroup>
      </Box>
      <svg width="800" height="450" viewBox="0 0 800 450">
        <BasketballCourt width={800} />
        <svg ref={courtSVG} className="courtSVG" width="800" height="450" />
      </svg>
    </Box>
  );
};
