import React, { useRef, useEffect, useState, useCallback } from "react";
import * as d3 from "d3";
import getTeamColor from "./userTeamColor";
import { BasketballCourt } from "./BasketballCourt";
import { Box, IconButton, Tooltip } from "@mui/material";
import PlayCircleOutlineRoundedIcon from "@mui/icons-material/PlayCircleOutlineRounded";
import PauseCircleOutlineRoundedIcon from "@mui/icons-material/PauseCircleOutlineRounded";
import RestartAltRoundedIcon from "@mui/icons-material/RestartAltRounded";
import EventRepeatOutlinedIcon from "@mui/icons-material/EventRepeatOutlined";
import useCombineData from "../useCombineData";
import axios from "axios";
import color from "./color";

const DrawPlayerVisualization = ({ width, playerData, T_type, isPlaying, currentStep, setCurrentStep, backList, setNewList }) => {
  const courtSVG = useRef();
  const isPlayingRef = useRef(isPlaying);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  const [newPlayerData, setNewPlayerData] = useState([]);

  useEffect(() => {
    if (playerData.length > 0) {
      setNewPlayerData(playerData);
    }
  }, [playerData]);

  useEffect(() => {
    const courtWidth = width;
    const courtHeight = (courtWidth * 50) / 94;

    const svg = d3.select(courtSVG.current);
    svg.selectAll("*").remove();

    const courtItem = svg.append("g").attr("class", "courtGroup");

    const playerIMGWidth = 100;

    const groups = courtItem
      .selectAll("g")
      .data(playerData, (d) => d.agent_id)
      .join("g")
      .attr(
        "transform",
        (d) =>
          `translate(${(d[T_type][1][0] * width) / 94}, ${(d[T_type][1][1] * width) / 94})`
      );

    groups.each(function (d, i) {
      const group = d3.select(this);

      const radius = i === 0 ? 10 : 25;

      const clipPathId = `clip-path-${i}`;
      if (d.agent_id !== -1) {
        group
          .append("clipPath")
          .attr("id", clipPathId)
          .append("circle")
          .attr("r", radius)
          .attr("cx", 0)
          .attr("cy", 0);

        if (i < 6) {
          group.raise();
        }
      } else {
        group.raise();
      }

      group
        .append("circle")
        .attr("class", i > 0 && i < 6 ? "off_circle" : "def_circle")
        .attr("r", radius)
        .style("fill", i === 0 ? "orange" : getTeamColor(d.teamID).color)
        .style("stroke-width", 3)
        .style("stroke", i === 0 ? "orange" : getTeamColor(d.teamID).color)
        .style("opacity", 0.9);

      group
        .append("text")
        .attr("class", i > 0 && i < 6 ? "off_text" : "")
        .attr("x", 0)
        .attr("y", 30)
        .style("font-size", 15)
        .style("text-anchor", "middle")
        .style("fill", "white")
        .attr("transform", "rotate(90)")
        .raise();

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
        .style("opacity", 0.7)
        .attr("transform", "rotate(90)");

      if (d.agent_id !== -1) {
        image.attr("clip-path", `url(#${clipPathId})`);
      }
    });
  }, [playerData, width, T_type]);

  useEffect(() => {
    if (!isPlaying) return;

    const courtGroup = d3.select(courtSVG.current).select(".courtGroup");
    const groups = courtGroup.selectAll("g");

    const drag = d3
      .drag()
      .on("start", function (event, d) {
        d3.select(this).raise();
      })
      .on("drag", function (event, d) {
        d3.select(this).attr("transform", `translate(${event.x}, ${event.y})`);
      })
      .on("end", function (event, d) {
        const newX = (event.x * 94) / width;
        const newY = (event.y * 94) / width;

        const existingIndex = backList.findIndex(
          (entry) => entry[0] === currentStep && entry[1] === d.player_index
        );
        if (existingIndex >= 0) {
          backList[existingIndex] = [currentStep, d.player_index, newX, newY];
        } else {
          backList.push([currentStep, d.player_index, newX, newY]);
        }

        setNewList([...backList]);
      });

    groups.call(drag);

    async function movePlayersSequentially(startStep) {
      let index = startStep;
      while (isPlayingRef.current) {
        if (
          newPlayerData.length === 0 ||
          !newPlayerData[0][T_type] ||
          index >= newPlayerData[0][T_type].length
        ) {
          setIsPlaying(false);
          break;
        }

        await new Promise((resolve) => {
          groups
            .transition()
            .duration(200)
            .ease(d3.easeLinear)
            .attr("transform", (d) => {
              if (index < d[T_type].length) {
                return `translate(${(d[T_type][index][0] * width) / 94}, ${(d[T_type][index][1] * width) / 94})`;
              }
              return `translate(${(d[T_type][0][0] * width) / 94}, ${(d[T_type][0][1] * width) / 94})`;
            })
            .on("end", resolve);

          d3.selectAll(".off_circle").style("fill", (d) =>
            color(d[T_type === "ghost_T" ? "ghost_qsq" : "real_qsq"][index])
          );

          d3.selectAll(".off_text")
            .text((d) => {
              const property = T_type === "ghost_T" ? "ghost_qsq" : "real_qsq";
              return parseFloat(d[property][index].toFixed(2));
            })
            .raise();
        });

        index++;
        setCurrentStep(index);

        if (index === newPlayerData[0][T_type].length) {
          setIsPlaying(false);
          setCurrentStep(1);
          break;
        }
      }
    }

    movePlayersSequentially(currentStep);
  }, [isPlaying, currentStep, newPlayerData, T_type, playerData, backList, width]);

  return <svg ref={courtSVG} className="courtSVG" width={width} height={(width * 50) / 94} />;
};

export const DrawPlayers = ({
  width,
  playerData,
  selectedDataFile,
  selectedCheckpoint,
}) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [newPlayerData, setNewPlayerData] = useState([]);
  const [newList, setNewList] = useState([]);
  const backList = [];

  useEffect(() => {
    if (playerData.length > 0) {
      setNewPlayerData(playerData);
      setCurrentStep(1);
    }
  }, [playerData]);

  const handlePlayPause = () => {
    setIsPlaying((prevIsPlaying) => !prevIsPlaying);
  };

  const handleRestart = () => {
    setIsPlaying(false);
    setCurrentStep(1);
    setIsPlaying(true);
  };

  const handleRerun = async () => {
    try {
      const response = await axios.post(
        "http://localhost:8080/api/update_data",
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

      if (response.data && response.data.error) {
        console.error("Error from server:", response.data.error);
        return;
      }

      const combinedData = useCombineData(response, "new");
      setNewPlayerData(combinedData);
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
      </Box>
      <Box sx={{ display: "flex", justifyContent: "space-around" }}>
        <Box>
          <h3>Ghost T</h3>
          <svg width={width} height={(width * 50) / 94}>
            <g transform="rotate(-90 200 200)">
              <BasketballCourt width={width} />
              <DrawPlayerVisualization
                width={width}
                playerData={newPlayerData}
                T_type="ghost_T"
                isPlaying={isPlaying}
                currentStep={currentStep}
                setCurrentStep={setCurrentStep}
                backList={backList}
                setNewList={setNewList}
              />
            </g>
          </svg>
        </Box>
        <Box>
          <h3>Real T</h3>
          <svg width={width} height={(width * 50) / 94}>
            <g transform="rotate(-90 200 200)">
              <BasketballCourt width={width} />
              <DrawPlayerVisualization
                width={width}
                playerData={newPlayerData}
                T_type="real_T"
                isPlaying={isPlaying}
                currentStep={currentStep}
                setCurrentStep={setCurrentStep}
                backList={backList}
                setNewList={setNewList}
              />
            </g>
          </svg>
        </Box>
      </Box>
    </Box>
  );
};
