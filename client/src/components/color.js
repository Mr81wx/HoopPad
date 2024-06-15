import { scaleQuantize, interpolateRgbBasis, scaleDiverging } from 'd3';


const color = (value) => {

    // const [dataState] = useDataState();
    const x0 = scaleQuantize()
        .range(['#00bbff', '#43c1ff', '#60c6ff', '#77ccff', '#8ad1ff', '#9bd7ff', '#abddff', '#bbe2ff', '#c9e8ff', '#d7eeff', '#e5f3ff', '#f2f9ff', '#ffffff', '#fff5ed', '#ffebdc', '#ffe1ca', '#ffd7b8', '#ffcda6', '#ffc394', '#ffb881', '#ffad6e', '#ffa15a', '#ff9544', '#ff892b', '#ff7b00']
        )
        .domain([-1, 1])

    // ['#550000', '#570000', '#5a0000', '#5d0000', '#600000', '#620000', '#650000', '#680000', '#6b0000', '#6d0000', '#700000', '#730000', '#760000', '#790000', '#7c0000', '#7f0000', '#810000', '#840000', '#870000', '#8a0000', '#8d0000', '#900000', '#930000', '#960000', '#990001', '#9c0003', '#9f0006', '#a20008', '#a5000b', '#a8000d', '#ab000f', '#ae0011', '#b10013', '#b40015', '#b70017', '#ba0019', '#bd001b', '#c0001d', '#c3001f', '#c60021', '#c90023', '#cc0025', '#cf0327', '#d10928', '#d30f2a', '#d5142b', '#d7182d', '#da1c2e', '#dc1f30', '#de2231', '#e02533', '#e22834', '#e52b36', '#e72d37', '#e93039', '#eb323a', '#ed353c', '#ef373d', '#f2393f', '#f43c40', '#f63e42', '#f84044', '#fb4345', '#fd4547', '#ff4748', '#ff4c4c', '#ff5150', '#ff5553', '#ff5a57', '#ff5e5a', '#ff625d', '#ff6660', '#ff6963', '#ff6d67', '#ff716a', '#ff746d', '#ff786f', '#ff7b72', '#ff7e75', '#ff8278', '#ff857b', '#ff887e', '#ff8b80', '#ff8e83', '#ff9186', '#ff9488', '#ff978b', '#ff9a8e', '#ff9d90', '#ffa093', '#ffa396', '#ffa698', '#ffa89b', '#ffab9d', '#ffaea0', '#ffb1a2', '#ffb3a5', '#ffb6a7', '#ffb9aa']


    const interpolate = interpolateRgbBasis(x0.range())

    const colorScale = scaleDiverging()
        .interpolator(interpolate)
        .domain([20, 50, 80])

    return colorScale(value)
}


export default color