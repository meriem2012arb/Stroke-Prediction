from json import load
from flask import Flask 
from flask import request
from flask import jsonify

import pickle


def load(filename):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)

app = Flask('predict_stroke')

dv = load('dv.bin')
model = load('model00.bin')


@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0, 1]
    stroke = y_pred >= 0.5

    result = {
        'stroke_probability': float(y_pred),
        'stroke': bool(stroke)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
    