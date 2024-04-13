import React, { useState, useEffect } from 'react';
import './App.css';
import { AppBar, Toolbar, Typography, Box } from '@mui/material';


function App() {
  // Initial example data
  const playerPositions = [
    { 'player_id': '123', 'x': 12, 'y': 20, 'group': 'offensive' },
    { 'player_id': '124', 'x': 12, 'y': 20, 'group': 'offensive' },
    { 'player_id': '123', 'x': 12, 'y': 20, 'group': 'offensive' },
    { 'player_id': '124', 'x': 12, 'y': 20, 'group': 'offensive' }
  ];

  const [currentPositions, setCurrentPositions] = useState(playerPositions);

  // useEffect(() => {
  //   fetch('/update_positions', {
  //     method: 'POST',
  //     headers: { 'Content-Type': 'application/json' },
  //     body: JSON.stringify({ "data": playerPositions })
  //   })
  //     .then((response) => response.json())
  //     .then((data) => {
  //       setCurrentPositions(data);
  //     })
  //     .catch((error) => {
  //       console.error('Error updating positions:', error);
  //     });
  // }, []);

  return (
    <div className="App">
      <AppBar position="absolute"
        sx={{
          width: "100%",
          backgroundColor: "#101010",
          height: {
            md: 60,
            lg: 65
          },
        }}
      >
        <Toolbar>
          <Typography variant="h4"
            className='appTitle'>
            Hoop Pad
          </Typography>
          <Box sx={{ flexGrow: 1 }} />

        </Toolbar>
      </AppBar>

      <Box sx={{ mt: 10 }}>
        {currentPositions.length > 0 && currentPositions.map(d => (
          <p key={d.player_id}>x: {d.x} y: {d.y}</p>
        ))}
      </Box>
    </div>
  );
}

export default App;
