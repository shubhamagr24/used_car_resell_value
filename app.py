from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    features = [x for x in request.form.values()]
    final_features = [list(features)]

    final_features[0]=final_features[0]//1000
    
    # Make prediction
    prediction = model.predict(pd.DataFrame(final_features,columns=['km_driven_in_K',
    'fuel',
    'transmission',
    'city',
    'model_year',
    'brand_name',
    'model_name',
    'registraton_validity'])
    )
    output = prediction[0]

    return render_template('index.html', prediction_text='Prediction: {} Lakhs'.format(round(output,2)))

if __name__ == "__main__":
    app.run(debug=True)