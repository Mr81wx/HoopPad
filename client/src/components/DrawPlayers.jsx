import { useRef, useEffect } from "react";
import * as d3 from "d3";
import getTeamColor from "./userTeamColor";

export const DrawPlayers = ({ width, playerData }) => {
  const courtSVG = useRef();
  useEffect(() => {
    const courtWidth = width; // Assuming 'width' is defined elsewhere as the width of the full court
    const courtHeight = (courtWidth * 50) / 94; // Height is computed to maintain the aspect ratio for a 94x50 court

    const svg = d3.select(courtSVG.current);
    svg.selectAll("*").remove(); // Clear the SVG to avoid duplicate content

    const courtItem = svg.append("g").attr("class", "courtGroup");

    const ghost_T = playerData.map((d) => d.ghost_T);

    const real_T = playerData.map((d) => d.real_T);

    // Define the circles representing players, positioned initially at the origin
    const circles = courtItem
      .selectAll("circle")
      .data(real_T)
      .join("circle")
      .attr("r", (d, i) => (i === 0 ? 5 : 10))
      .attr(
        "transform",
        (d) => `translate(${(d[0][0] * width) / 94}, ${(d[0][1] * width) / 94})`
      ) 
      .style("fill", (d, i) => (i === 0 ? "orange" : i > 5 ? "red" : "blue"))
      .style("opacity", 0.7);

    async function movePlayersSequentially() {
      for (let index = 0; index < 40; index++) {
        await new Promise((resolve) => {
          circles
            .transition()
            .duration(200)
            .attr(
              "transform",
              (d) =>
                `translate(${(d[index][0] * width) / 94}, ${
                  (d[index][1] * width) / 94
                })`
            )
            .end()
            .then(resolve);
        });
      }
    }

    movePlayersSequentially();
  }, [playerData, width]);

  return <svg ref={courtSVG} className="courtSVG" />;
};
