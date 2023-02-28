from flask import Flask, render_template, request
from model import load, prediksi

app = Flask(__name__)

# load model
load()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Team')
def team():
    return render_template('team.html')


@app.route('/predict', methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
        # Menangkap data yang diinput user melalui form
        nama_lengkap = str(request.form['nama_lengkap'])
        nilai = float(request.form['nilai'])
        pendapatan_keluarga = int(request.form['pendapatan_keluarga'])
        jumlah_tanggungan = int(request.form['jumlah_tanggungan'])
        # Melakukan prediksi menggunakan model yang telah dibuat
        data = [[nilai, pendapatan_keluarga, jumlah_tanggungan]]
        prediction_result, confidence = prediksi(data)
        return render_template('prediction.html', nama_mahasiswa=nama_lengkap, hasil_prediksi=prediction_result, 
                                nilai_kepercayaan=confidence)
    else:
        return render_template('prediction.html')

if __name__ == "__main__":
    app.run(debug=True)