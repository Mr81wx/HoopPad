import time
from flask import Flask, request, jsonify
import sys

from Model.geometry import velocity
from App.draw_cycle import draw_cycle

app = Flask(__name__)

# Endpoint to update defensive player positions
@app.route('/update_positions', methods=['POST'])

def update_positions():
    data = request.get_json()
    new_positions = calculate_defensive_positions(data)
    return jsonify(new_positions)

def calculate_defensive_positions(data):
   
    return {"defenders": data} 

if __name__ == '__main__':
    app.run(debug=True)