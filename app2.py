from flask import Flask, render_template, request
import pickle as pkl
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
with open('mm1.pkl','rb') as file:
    model = pkl.load(file)

@app.route('/')
def iris():
    return render_template('iris.html')

@app.route('/predict', methods=['POST'])
def predict():

    s_len = request.form.get('sepal_length')
    s_wid = request.form.get('sepal_width')
    p_len = request.form.get('petal_length')
    p_wid = request.form.get('petal_width')

    clean_data = [float(s_len), float(s_wid), float(p_len), float(p_wid)]

    feature_names = [
        'sepal length in cm',
        'sepal width in cm',
        'petal length in cm',
        'petal width in cm'
    ]

    ex1 = pd.DataFrame([clean_data], columns=feature_names)

    prediction = model.predict(ex1)

    return render_template('iris.html', prediction_text=f'Flower species is {prediction[0]}')

if __name__ == "__main__":
    app.run()