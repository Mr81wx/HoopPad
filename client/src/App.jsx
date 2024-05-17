import { useState, useEffect } from "react";

import "./App.css";
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
} from "@mui/material";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import axios from "axios";
import { DrawPlayers } from "./components/DrawPlayers";
import useCombineData from './useCombineData'


function App() {
  const [count, setCount] = useState(0);
  const [dataFiles, setDataFiles] = useState([]);
  const [checkpointFiles, setCheckpointFiles] = useState([]);
  const [selectedDataFile, setSelectedDataFile] = useState("");
  const [selectedCheckpoint, setSelectedCheckpoint] = useState("");

  const [playerData, setPlayerData] = useState([]);

  const fetchAPI = async () => {
    // const response = await axios.get("http://localhost:8080/api/ghostT");

    try {
      const dataResponse = await axios.get("http://localhost:8080/select_data");
      setDataFiles(
        dataResponse.data.sort((a, b) => a.label.localeCompare(b.label))
      );

      const checkpointResponse = await axios.get(
        "http://localhost:8080/select_checkpoints"
      );
      console.log(checkpointResponse.data);
      setCheckpointFiles(
        checkpointResponse.data.sort((a, b) => a.label.localeCompare(b.label))
      );
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  const handleSubmit = async () => {
    try {
      const response = await axios.get(`http://localhost:8080/api/get_data`, {
        params: {
          dataFile: selectedDataFile,
          checkpointFile: selectedCheckpoint,
        },
      });

      console.log("All response data:", response.data.agent_IDs);

      const combinedData = useCombineData(response)

      console.log("Combined data:", combinedData);
      setPlayerData(combinedData);
    } catch (error) {
      console.error("Error submitting selections:", error);
    }
  };

  const theme = createTheme({
    palette: {
      mode: "dark",
    },
  });

  useEffect(() => {
    fetchAPI();
  }, []);

  return (
    <>
      <AppBar
        position="absolute"
        sx={{
          width: "100%",
          backgroundColor: "#101010",
          height: {
            md: 60,
            lg: 65,
          },
        }}
      >
        <Toolbar>
          <h2>HoopPad</h2>
          <Box sx={{ flexGrow: 1 }} />
        </Toolbar>
      </AppBar>

      <div className="card">
        <ThemeProvider theme={theme}>
          {
            <div className="card">
              <FormControl fullWidth>
                <InputLabel id="data-select-label">Select Data</InputLabel>
                <Select
                  labelId="data-select-label"
                  value={selectedDataFile}
                  label="Select Data"
                  onChange={(e) => setSelectedDataFile(e.target.value)}
                >
                  {dataFiles.map((file) => (
                    <MenuItem key={file.id} value={file.id}>
                      {file.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <FormControl fullWidth>
                <InputLabel id="checkpoint-select-label">
                  Select Checkpoint
                </InputLabel>
                <Select
                  labelId="checkpoint-select-label"
                  value={selectedCheckpoint}
                  label="Select Checkpoint"
                  onChange={(e) => setSelectedCheckpoint(e.target.value)}
                >
                  {checkpointFiles.map((file) => (
                    <MenuItem key={file.id} value={file.id}>
                      {file.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <Button
                variant="contained"
                color="primary"
                onClick={handleSubmit}
                style={{ marginTop: "20px" }}
              >
                Submit Selections
              </Button>
            </div>
          }
          <div className="card">
           
          <DrawPlayers 
          width={800}
          playerData={playerData}
          selectedDataFile = {selectedDataFile}
          selectedCheckpoint = {selectedCheckpoint}
           />
            {/* <svg width="800" height="450" viewBox="0 0 800 450">
              <BasketballCourt width={800} />
              {playerData.length > 0 && (
                <DrawPlayers width={800} playerData={playerData} />
              )}
            </svg> */}
          </div>
        </ThemeProvider>
      </div>
    </>
  );
}

export default App;
