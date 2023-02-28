import pickle

# global variable
global model, scaler

def load():
    global model, scaler
    model = pickle.load(open('decision-tree.pkl', 'rb'))
    scaler = pickle.load(open('scalar.pkl', 'rb'))

def prediksi(data):
    data = scaler.transform(data)
    prediksi = int(model.predict(data))
    nilai_kepercayaan = model.predict_proba(data).flatten()
    nilai_kepercayaan = max(nilai_kepercayaan) * 100
    nilai_kepercayaan = round(nilai_kepercayaan)

    if prediksi == 0:
        hasil_prediksi = "TIDAK LOLOS"
    else:
        hasil_prediksi = "LOLOS"
    return hasil_prediksi, nilai_kepercayaan