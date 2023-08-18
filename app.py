from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)
script_dir =os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_dir, 'diabity_joblib.joblib')
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)


@app.route('/cal', methods=['POST'])
def cal():
    data_a = float(request.form['a'])
    data_b = float(request.form['b'])
    data_c = float(request.form['c'])
    data_d = float(request.form['d'])
    data_e = float(request.form['e'])
    data_f = float(request.form['f'])
    data_g = float(request.form['g'])
    data_h = float(request.form['h'])
    arr = np.array([[data_a, data_b, data_c, data_d, data_e, data_f, data_g, data_h]])
    pred = model.predict(arr)
    if pred == 0:
        result = "NO"
    else:
        result = "YES"
    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
