from flask import Flask, redirect, request, render_template, jsonify
import pickle
import numpy as np

app=Flask(__name__)
model= pickle.load(open('ten_year_chd_model', 'rb'))
scaler= pickle.load(open('scaler', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features= [int(x) for x in request.form.values()]
    print(features)
    final_features= [np.array(features)]
    scaled_features= scaler.transform(final_features)
    prediction= model.predict(scaled_features)

    output=" "
    if prediction==1:
        output="Yes"
    else:
        output='No'

    return render_template('index.html', prediction_text= "Possibility of Heart Disease in next 10 years: {}".format(output))

if __name__=='__main__':
    app.run(debug=True)
