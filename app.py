
from flask import Flask,request, render_template, Response,Markup
import cv2
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np
import datetime
import os
import tensorflow as tf
import base64
import PIL.Image as Image
from utils.disease import disease_dic 

global model



app = Flask(__name__)

class_names = ['Potato___Early_blight',
              'Potato___Late_blight',
              'Potato___healthy']



model = load_model('potatoes.h5')

model.make_predict_function()

try:
    os.mkdir('static/shot')
except OSError as error:
    pass

try:
    os.mkdir('static/upload')
except OSError as error:
    pass


def predict_label(image):
    

    image = np.array(Image.open(image).convert("RGB").resize((256, 256)))


    image = image/255 

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return {"class": predicted_class, "confidence": confidence}


@app.route("/")
def main():
    return render_template("index.html")





@app.route('/jsTakePic',methods=['POST','GET'])
def jsTakePic():
    return render_template('jsTakePic.html')

    
  
@app.route("/shot",methods = ['GET', 'POST'])
def shot():
    if request.method == 'POST':
        
        data = request.form['image']
        print(data)

                 
        data = data.split(',')[1]

        print(data)
       
  
        # Using bytes(str, enc)
        # convert string to byte 
        res = bytes(data, 'utf-8')
        
        # print result
        now = datetime.datetime.now()
        img_path2 = os.path.sep.join(['static/shot', "shot_{}.jpg".format(str(now).replace(":",''))])	

       

        with open(img_path2, "wb") as fh:
            fh.write(base64.decodebytes(res))

        
        result2 = predict_label(img_path2)
        prediction = Markup(str(disease_dic[prediction]))
        if (result2["confidence"]>=0.6):
            return render_template("output.html", prediction=result2["class"],confidence = result2["confidence"],img_path = img_path2,rd = prediction)
        else:
            return render_template("output.html",msg = "No Match Found")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img1 = request.files['my_image']
        img_path1 = "static/upload" + img1.filename	
        img1.save(img_path1)
        result1 = predict_label(img_path1)
        prediction = Markup(str(disease_dic[prediction]))

        if (result1["confidence"]>=0.6):
            return render_template("output.html", prediction=result1["class"],confidence = result1["confidence"],img_path = img_path1,rd = prediction)

        else:
            return render_template("output.html",msg = "No Match Found")



if __name__ =='__main__':
    app.run(debug = True)