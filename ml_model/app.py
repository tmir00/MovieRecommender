from model import run_model
from flask import Flask, request, jsonify

app = Flask(__name__)


# Handle Post Request
@app.route('/predict', methods=['POST'])
def predict():
    """
    Obtain user id from post request
    :return:
    """
    # Check if the request contains JSON data
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    # Check for POST data.
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

    # Check for 'user_id' in the JSON data
    user_id = data.get('user_id')
    if user_id is None:
        return jsonify({"error": "Missing 'user_id' in request data"}), 400

    # Run the ML model, respond with error if it fails.
    try:
        prediction = run_model(user_id)
    except Exception as e:
        return jsonify({"error": "An error occurred during prediction", "details": str(e)}), 500

    # If we are here, everything worked as planned, extract ML model response
    response = {
        'movies': prediction[0],
        'account': prediction[1]
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
