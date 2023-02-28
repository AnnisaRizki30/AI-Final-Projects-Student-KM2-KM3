# Importing essential libraries
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import skimage
import skimage.transform
from io import BytesIO
import matplotlib.pyplot as plt
import base64
from PIL import Image, ImageFile
import pickle

classifier = pickle.load(open('model_diabetes_rf.pkl', 'rb'))
scaler = pickle.load(open('robust-scaler.pkl', 'rb'))
model = load_model('model_pnuemonia.h5')

app = Flask(__name__)


@app.route('/')
def home():
	return render_template('index.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    global preg, glucose, bp, st, insulin, bmi, dpf, age, gp, ip, ob1, ob2, ob3, ow, uw, is_normal, gs_normal, gs_over, gs_high
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']
        gp =  request.form['glucose_pregnancies']
        ip =  request.form['insuline_pregnancies']
        ob1 = request.form['obesity1']
        ob2 = request.form['obesity2']
        ob3 = request.form['obesity3']
        ow = request.form['overweight']
        uw = request.form['underweight']
        is_normal = request.form['insulin_normal']
        gs_normal = request.form['glucose_normal']
        gs_over = request.form['glucose_overweight']
        gs_high = request.form['glucose_high']
        data = [[preg, glucose, bp, st, insulin, bmi, dpf, age, gp, ip]]
        data = pd.DataFrame(data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'GP', 'IP'])
        cols = data.columns
        index = data.index
        data = scaler.transform(data)
        X = pd.DataFrame(data, columns = cols, index = index)
        categorical_var = [[ob1, ob2, ob3, ow, uw, is_normal, gs_normal, gs_over, gs_high]]
        categorical_data = pd.DataFrame(categorical_var, columns=['Obesitas1', 'Obesitas2', 'Obesitas3', 'Overweight',
                                                                  'Underweight', 'InsulinNormal', 'GlucoseNormal', 
                                                                  'GlucoseOverweight', 'GlucoseHigh'])
        final_data = pd.concat([X,categorical_data], axis = 1)
        prediksi = int(classifier.predict(final_data))
        nilai_kepercayaan = classifier.predict_proba(final_data).flatten()
        nilai_kepercayaan = max(nilai_kepercayaan) * 100
        nilai_kepercayaan = round(nilai_kepercayaan)
        print(nilai_kepercayaan)
        if prediksi == 1:
            hasil_prediksi = "Danger"
        else:
            hasil_prediksi = "Safe"
        print(hasil_prediksi)
        return render_template('diabetes.html', hasil_prediksi=hasil_prediksi, nilai_kepercayaan=nilai_kepercayaan)


@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumonia():
    return render_template('pneumonia.html')

def predict_image(image_data, model):
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    org_img = Image.open(BytesIO(image_data.read()))
    org_img.load()
    img = org_img.resize((150,150), Image.ANTIALIAS)
    img = image.img_to_array(img)
    org_img = image.img_to_array(org_img)
    img = np.expand_dims(img, axis = 0)
    result = model.predict(img)
    if result[0][0] == 0:
        prediction = 'Negative (Normal)'
    else:
        prediction = 'Positive (Pneumonia)'
    return org_img, prediction


@app.route("/predict_pneumonia", methods = ['GET', 'POST'])
def pneumoniapredict():
    global org_img, pred
    if request.method == 'POST':
        file = request.files['file']
        org_img, pred = predict_image(file, model)
        
        img_x=BytesIO()
        plt.imshow(org_img[:,:,0]/255.0)
        plt.savefig(img_x,format='png')
        plt.close()
        img_x.seek(0)
        plot_url=base64.b64encode(img_x.getvalue()).decode('utf8')

    return render_template('pneumonia.html', pred = pred, plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
