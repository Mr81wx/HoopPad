/* --------------------------- draw an empty court -------------------------- */

import { useRef, useEffect } from 'react';
import { select } from 'd3';

export const BasketballCourt = ({
    width,
}) => {
    const courtSVG = useRef()
    useEffect(() => {

        const courtWidth = width;
        const courtHeight = courtWidth / 94 * 50


        const courtItem = select(courtSVG.current)
            .append('g')
            .attr('class', 'courtGroup')
        // .attr('transform', 'translate(' + (courtIndex === 2 ? 75 : 200) + ',' + (courtIndex === 2 ? dimensions.margin.top - 50 : dimensions.margin.top) + ')')

        courtItem.append('svg:image')
            // .attr('xlink:href', 'https://raw.githubusercontent.com/fuyuGT/CS7450-data/main/fullcourt.svg')
            .attr('xlink:href', 'https://raw.githubusercontent.com/fuyuGT/CS7450-data/main/halfcourt.svg')

            // .attr('transform', 'translate(' + (dimensions.courtWidth / 2 + dimensions.courtWidth / 94 * 3) + ',' + 0 + ')')
            .style('opacity', 0.5)
            .attr('width', courtWidth)
            .attr('height', courtHeight)
            .attr('transform', 'translate(' + (-courtWidth/4) + ',' + 0 + ')')

    }, [])

    return (
        <g ref={courtSVG} className='NBAcourt' />

    )
}
