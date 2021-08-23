import numpy as np
from flask import Flask,request,jsonify, render_template
import pickle

from os import environ


 
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def hello():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]

    final_features = [np.array(int_features)]
    prediction=model.predict(final_features)
    #output = '{0:.{1}{f}'.format(prediction[0][1],2)

    if prediction>0.7:
        return render_template('index.html',pred=f'Applicant is eligible for loan')
    else:
        return render_template('index.html',pred=f'Sorry, Applicant is not eligible for the loan')



if __name__ == '__main__':
    app.run(debug=True)
