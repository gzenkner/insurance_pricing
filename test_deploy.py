import requests
import json
from flask import Flask, request, jsonify
import pickle
import numpy as np


# Define the user input data
# user_input = {
#     'age': 20,
#     'sex': 0,
#     'bmi': 20,
#     'children': 0,
#     'smoker': 0,
#     'region': 1
# }
user_input = {
    'age': 20,
    'sex': 0,
    'bmi': 20,
    'children': 0,
    'smoker': 1,
    'region': 1
}

# Define the URL of your Flask API
url = 'http://127.0.0.1:5000/predict'  # Make sure your Flask server is running

# Send a POST request with the user input as JSON
response = requests.post(url, json=user_input)

# Check the response
if response.status_code == 200:
    result = response.json()
    print("Cost of insurance premium in USD:", round(result["prediction"], 0))
else:
    print("Request failed with status code:", response.status_code)
