import time
from flask import Flask, request, jsonify
import sys

# from App.defense_generator import some_function
# from Data.dataload import another_function

app = Flask(__name__)

@app.route('/generate-def')
def generate_defense():
    try:
        possession_path = 'api/Data/object_0.pkl'
        ckp_path = 'api/Checkpoints/V3_prompt+def_loss.ckpt'
        states_batch, agents_batch_mask, states_padding_batch, states_hidden_batch, agent_ids_batch, team_ids_batch = load_possession(possession_path)
        model = load_model(ckp_path)
        real_T, ghost_T = def_gen(model, states_batch, agents_batch_mask, states_padding_batch, states_hidden_batch, agent_ids_batch, team_ids_batch)
        return jsonify(real_T.tolist(), ghost_T.tolist())
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)