# run from anaconda command prompt only
# in VS code numpy as issue
from flask import Flask,request,jsonify
import pickle

app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello World'
@app.route('/predict')
def predict():
    posted_data=request.get_json()
    sepal_length = posted_data['sepal_length']
    sepal_width = posted_data['sepal_width']
    petal_length = posted_data['petal_length']
    petal_width = posted_data['petal_width']
    model = pickle.load(open("knn_model.pkl", 'rb'))
    prediction=model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    if prediction == 0:
        predicted_class = 'Iris-setosa'
    elif prediction == 1:
        predicted_class = 'Iris-versicolor'
    else:
        predicted_class = 'Iris-virginica'

    return jsonify({
        'Prediction': predicted_class
    })
    
if __name__ == '__main__':
    app.run(debug=True)