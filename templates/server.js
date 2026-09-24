const express = require("express");
const { spawn } = require("child_process");
const cors = require("cors");

const app = express();
app.use(express.json()); // For parsing JSON data
app.use(cors()); // Allow cross-origin requests

app.post("/predict", (req, res) => {
  const inputData = req.body; // Get input data from the request body

  console.log("Received data:", inputData); // Log the received data

  const pythonProcess = spawn("python3", [
    "predict.py",
    JSON.stringify(inputData),
  ]);

  pythonProcess.stdout.on("data", (data) => {
    console.log("Python stdout:", data.toString()); // Log the output from Python

    const prediction = parseFloat(data.toString());
    if (isNaN(prediction)) {
      console.error("Invalid prediction output:", data.toString());
      res.status(500).json({ error: "Invalid prediction output" });
    } else {
      res.json({ mileage: prediction }); // Return mileage in JSON format
    }
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error("Python stderr:", data.toString()); // Log errors from Python
    res.status(500).json({ error: "Prediction failed" });
  });

  pythonProcess.on("exit", (code) => {
    if (code !== 0) {
      console.error(`Python process exited with code ${code}`);
      res.status(500).json({ error: "Prediction failed" });
    }
  });
});

app.listen(5000, () => {
  console.log("Server running on port 5000");
});
