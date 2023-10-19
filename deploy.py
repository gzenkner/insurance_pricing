from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('xgboost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.get_json()
    prediction = model.predict(np.array([user_input['age'], user_input['sex'], user_input['bmi'],
                                         user_input['children'], user_input['smoker'], user_input['region']]).reshape(1, -1))

    prediction = float(prediction[0])
    response = {
        "prediction": prediction
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
