from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    age = data.get('age')
    gender = data.get('gender')
    occupation = data.get('occupation')
    favoritemovie1 = data.get('favoritemovie1')
    favoritemovie2 = data.get('favoritemovie2')
    favoritemovie3 = data.get('favoritemovie3')

    prediction = model(age, gender, occupation, favoritemovie1, favoritemovie2, favoritemovie3)

    response = {
        'prediction': prediction
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
