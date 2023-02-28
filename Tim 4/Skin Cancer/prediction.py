from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageFile
from io import BytesIO

kincer_model=load_model('model/best_model_cutix.h5')

def preprocess(file):
    ImageFile.LOAD_TRUNCATED_IMAGES =False
    org_img=Image.open(BytesIO(file.read()))
    org_img.load()
    img=org_img.resize((28,28), Image.ANTIALIAS)

    img=image.img_to_array(img)
    org_img=image.img_to_array(org_img)
    return org_img, np.expand_dims(img,axis=0)

def classify_skin_cancer(img):
    kincer_dict={}
    kincer_dict['akiec']=round(kincer_model.predict(img)[0][0]*100,4)
    kincer_dict['bcc']=round(kincer_model.predict(img)[0][1]*100,4)
    kincer_dict['bkl']=round(kincer_model.predict(img)[0][2]*100,4)
    kincer_dict['df']=round(kincer_model.predict(img)[0][3]*100,4)
    kincer_dict['nv']=round(kincer_model.predict(img)[0][4]*100,4)
    kincer_dict['vasc']=round(kincer_model.predict(img)[0][5]*100,4)
    kincer_dict['mel']=round(kincer_model.predict(img)[0][6]*100,4)

    for value in kincer_dict:
     if kincer_dict[value]<=0.001:
        kincer_dict[value]=0.00

    return kincer_dict