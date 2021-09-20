from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np


model_filename = "Pickle_SVM_Model.pkl"
vectorizer_filename = 'Pickle_vectorizer.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

with open(vectorizer_filename, 'rb') as file:
    vectorizer = pickle.load(file)


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    JobTitle = request.form['JobTitle']
    print(JobTitle)
    JobTitle = np.array([JobTitle])
    print(JobTitle)

    sample = vectorizer.transform(JobTitle).toarray()

    preds = model.predict(sample)
    prediction = preds[0]
    return render_template('pred.html', data=prediction)


@app.route('/api/v1/predict/<string:text>', methods=['GET'])
def predict_api(text):

    JobTitle = text
    JobTitle = np.array([JobTitle])

    sample = vectorizer.transform(JobTitle).toarray()

    preds = model.predict(sample)
    prediction = preds[0]

    response = {
        "text": text,
        "prediction": prediction
    }
    return jsonify(response)


app.run(port=5000,   debug=True)
