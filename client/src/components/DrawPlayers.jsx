import { useRef, useEffect } from "react";
import * as d3 from "d3";

export const DrawPlayers = ({ width, playerData }) => {
  const courtSVG = useRef();
  useEffect(() => {
    const courtWidth = width;
    const courtHeight = (courtWidth / 94) * 50;

    const svg = d3.select(courtSVG.current);
    svg.selectAll("*").remove(); // Clear the SVG to avoid duplicate content

    const courtItem = svg.append("g").attr("class", "courtGroup");

    const circles = courtItem
      .selectAll("circle")
      .data(playerData)
      .join("circle")
      .attr("r", 10)
      .attr("cx", (d) => d[0][0] * 100)
      .attr("cy", (d) => d[0][1] * 100)
      .style("fill", (d,i)=> i == 0? 'yellow' : i>5?'red':'blue')
      .style("opacity", 0.7);

    function movePlayers(frameIndex) {
      circles
        .transition()
        .duration(2000) // Duration of transition between points
        .ease(d3.easeLinear)
        .attr("cx", (d) => d[frameIndex][0]*100)
        .attr("cy", (d) => d[frameIndex][1]*100);
    }

    for (let index = 0; index < 100; index++) {
      movePlayers(index);
    }

    // Add a circle to represent the player
    // const player = courtItem.append("circle");

    // player
    //   .attr("cx", playerData[0][0])
    //   .attr("cy", playerData[0][1])
    //   .attr("r", 5) // Radius of the circle
    //   .style("fill", "red");

    // // Function to move the circle through each position
    // function movePlayer(index) {
    //   player
    //     .transition()
    //     .duration(10) // Duration of transition between points
    //     .ease(d3.easeLinear)
    //     .attr("cx", playerData[index][0])
    //     .attr("cy", playerData[index][1])
    //     // .on("end", () => {
    //     //   // Call the next move after the current one finishes
    //     //   if (index < playerData.length - 1) {
    //     //     movePlayer(index + 1);
    //     //   }
    //     // });
    // }

    // // Start the animation
    // if (playerData.length > 1) {
    //   movePlayer(1); // Start from the second point
    // }
  }, [playerData]);

  return <svg ref={courtSVG} className="courtSVG" />;
};
