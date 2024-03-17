from model import run_model
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    user_id = data.get('age')

    prediction = run_model(user_id)

    response = {
        'prediction': prediction[1],
        'account': prediction[0]
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
