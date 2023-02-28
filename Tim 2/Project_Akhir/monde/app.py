from process import preparation, botResponse
from flask import Flask, render_template, request, jsonify, Response
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.models import load_model
from io import BytesIO
import matplotlib.pyplot as plt
import base64
from PIL import Image, ImageFile
import cv2

# download nltk
preparation()

#Start Chatbot
app = Flask(__name__)
app.static_folder = 'static'

model = load_model('model/model-monkeypox.h5')

camera = cv2.VideoCapture(0)

@app.route("/")
def home():
    return render_template("page_awal.html")

@app.route("/menu")
def menu():
    return render_template("menu.html")

@app.route("/about")
def about():
    return render_template("about_us.html")

@app.route("/image_detection")
def image_detection():
    return render_template("image_detection.html")

@app.route("/realtime_detection")
def realtime_detection():
    return render_template("realtime_detection.html")

@app.route('/Predict', methods=["GET", "POST"])
def predict():
    text = request.get_json().get("message")
    response = botResponse(text)
    message = {"answer": response}
    return jsonify(message)


def predict_image(image_data, model):
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    org_img = Image.open(BytesIO(image_data.read()))
    org_img.load()
    img = org_img.resize((224,224), Image.ANTIALIAS)
    img = image_utils.img_to_array(img)
    org_img = image_utils.img_to_array(org_img)
    img = np.expand_dims(img, axis = 0)
    result = model.predict(img)
    print(result)
    if result >= 0.5:
        string_prediction = 'MonkeyPox'
        prediction = ('{:.2%} percent confirmed MonkeyPox case'.format(result[0][0]))
    else:
        string_prediction = 'Others'
        prediction = ('{:.2%} percent confirmed Other case'.format(1-result[0][0]))
    return org_img, prediction, string_prediction


@app.route("/predict_monkeypox", methods = ['GET', 'POST'])
def monkeypox_predict():
    global org_img, pred, string_pred, plot_url
    if request.method == 'POST':
        file = request.files['file']
        org_img, pred, string_pred = predict_image(file, model)
        
        img_x=BytesIO()
        plt.imshow(org_img/255.0)
        plt.savefig(img_x,format='png')
        plt.close()
        img_x.seek(0)
        plot_url=base64.b64encode(img_x.getvalue()).decode('utf8')

    return render_template('image_detection.html', pred = pred, string_pred = string_pred, plot_url=plot_url)

def gen_frames(): 
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        frame = cv2.flip(frame, 1)
        if not success:
            break
        else:
            print("[INFO] loading and preprocessing image...")
            image = Image.fromarray(frame, 'RGB')
            image = image.resize((224,224))
            image = image_utils.img_to_array(image)
            image = np.expand_dims(image, axis=0)

            #classify the image
            print("[INFO] classifying image...")
            preds = model.predict(image)
            if preds >= 0.5:
                prediction = ('{:.2%} percent confirmed MonkeyPox case'.format(preds[0][0]))
            else:
                prediction = ('{:.2%} percent confirmed Other case'.format(1-preds[0][0]))

            cv2.putText(frame, "Label: {}".format(prediction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (209, 80, 0, 255), 2)

            if cv2.waitKey(5) & 0xFF == 27:
                break

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
                
@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)