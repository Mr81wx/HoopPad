import time
from flask import Flask, jsonify, request
from flask_cors import CORS
from App.defense_generator import *
import json
import numpy as np
from numpyencoder import NumpyEncoder
import logging

sys.path.append("../server")


DATA_DIR = "Data"
CHECKPOINTS_DIR = "Checkpoints"

app = Flask(__name__)
cors = CORS(app, origins="*")


@app.route("/select_data")
def select_data():
    data_files = [
        {"id": f, "label": f} for f in os.listdir(DATA_DIR) if f.endswith(".pkl")
    ]
    print(data_files)
    return jsonify(data_files)


@app.route("/select_checkpoints")
def select_checkpoints():
    checkpoint_files = [
        {"id": f, "label": f}
        for f in os.listdir(CHECKPOINTS_DIR)
        if f.endswith(".ckpt")
    ]
    return jsonify(checkpoint_files)


@app.route("/api/get_data")
def get_data():
    data_file = request.args.get("dataFile")
    checkpoint_file = request.args.get("checkpointFile")

    if not data_file or not checkpoint_file:
        logging.error("Required file parameters are missing.")
        return jsonify({"error": "Missing required parameters"}), 400

    possession_path = os.path.join(DATA_DIR, data_file)
    ckp_path = os.path.join(CHECKPOINTS_DIR, checkpoint_file)

    try:
        results = get_results(possession_path, ckp_path)

        serialized_results = {
            "ghost_T": serialize_tensor(results["ghost_T"].clone().detach()),
            "real_T": serialize_tensor(results["real_T"].clone().detach()),
            "team_IDs": results["team_IDs"].tolist(),
            "agent_IDs": results["agent_IDs"].tolist(),
            "player_detail": results["player_detail"].tolist(),
            "real_QSQ": results["real_QSQ"].tolist(),
            "ghost_QSQ": results["ghost_QSQ"].tolist(),
        }

        return jsonify(serialized_results)
    except Exception as e:
        logging.error(f"Failed to process tensor data: {str(e)}")
        return jsonify({"error": "Failed to process tensor data"}), 500


@app.route("/api/update_data", methods=["POST"])
def update_data():
    print("request", request.args, request.json)
    data_file = request.args.get("dataFile")
    checkpoint_file = request.args.get("checkpointFile")
    back_list = request.json.get("backList", [])

    print(back_list)

    if not data_file or not checkpoint_file or not back_list:
        logging.error("Required parameters are missing.")
        return jsonify({"error": "Missing required parameters"}), 400

    possession_path = os.path.join(DATA_DIR, data_file)
    ckp_path = os.path.join(CHECKPOINTS_DIR, checkpoint_file)

    try:
        results = update_results(back_list, possession_path, ckp_path)
        serialized_results = {
            "ghost_T": serialize_tensor(results["ghost_T"].clone().detach()),
            "real_T": serialize_tensor(results["real_T"].clone().detach()),
            "team_IDs": results["team_IDs"].tolist(),
            "agent_IDs": results["agent_IDs"].tolist(),
            "player_detail": results["player_detail"].tolist(),
            "real_QSQ": results["real_QSQ"].tolist(),
            "ghost_QSQ": results["ghost_QSQ"].tolist(),
        }

        return jsonify(serialized_results)
    except Exception as e:
        logging.error(f"Failed to process tensor data: {str(e)}")
        return jsonify({"error": "Failed to process tensor data"}), 500


def serialize_tensor(tensor):
    return {"data": tensor.tolist(), "dtype": str(tensor.dtype), "shape": tensor.shape}


if __name__ == "__main__":
    app.run(debug=True, port=8080)
