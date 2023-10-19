from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('xgboost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# provides a route that responds to HTTP request methods (POST), the function will be called when a POST request is made to the '/predict' URL. e.g., 127.0.0.1.5000/predict
@app.route('/predict', methods=['POST'])
def predict():
    # predict function will be executed when a post request is made to 127.0.0.1.5000/predict

    # get the json data from HTTP REQUEST
    user_input = request.get_json()
    prediction = model.predict(np.array([user_input['age'], user_input['sex'], user_input['bmi'],
                                         user_input['children'], user_input['smoker'], user_input['region']]).reshape(1, -1))

    prediction = float(prediction[0])
    response = {
        "prediction": prediction
    }
    # returns a JSON response to the client that made the POST request
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
