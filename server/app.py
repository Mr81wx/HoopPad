import time
from flask import Flask, jsonify, request
from flask_cors import CORS
from App.defense_generator import *
import json
import numpy as np
from numpyencoder import NumpyEncoder
import logging


DATA_DIR = '/Users/yufu/Documents/Code/HoopPad/server/Data'
CHECKPOINTS_DIR = '/Users/yufu/Documents/Code/HoopPad/server/Checkpoints'

app = Flask(__name__)
cors = CORS(app, origins='*')


@app.route('/select_data')
def select_data():
    data_files = [{"id": f, "label": f} for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]
    print(data_files)
    return jsonify(data_files)

@app.route('/select_checkpoints')
def select_checkpoints():
    checkpoint_files = [{"id": f, "label": f} for f in os.listdir(CHECKPOINTS_DIR) if f.endswith('.ckpt')]
    return jsonify(checkpoint_files)


@app.route('/api/ghostT')
def ghostT():
    data_file = request.args.get('dataFile')
    checkpoint_file = request.args.get('checkpointFile')

    if not data_file or not checkpoint_file:
        logging.error("Required file parameters are missing.")
        return jsonify({'error': 'Missing required parameters'}), 400

    possession_path = os.path.join('/Users/yufu/Documents/Code/HoopPad/server/Data', data_file)
    ckp_path = os.path.join('/Users/yufu/Documents/Code/HoopPad/server/Checkpoints', checkpoint_file)

    try:
        ghost_list = get_results(possession_path, ckp_path).clone().detach()
        serialized_tensor = serialize_tensor(ghost_list)
        return jsonify({'ghost_T': serialized_tensor})
    except Exception as e:
        logging.error(f"Failed to process tensor data: {str(e)}")
        return jsonify({'error': 'Failed to process tensor data'}), 500
    

def serialize_tensor(tensor):
    return {
        'data': tensor.tolist(),  
        'dtype': str(tensor.dtype), 
        'shape': tensor.shape
    }


if __name__ == '__main__':
    with app.app_context():
        print(select_data())
    app.run(debug=True, port=8080)