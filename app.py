from flask import Flask, render_template, url_for, request, send_file, redirect
from werkzeug.utils import secure_filename

import sys,os,glob,re
import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from gevent.pywsgi import WSGIServer


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

new_model = tf.keras.models.load_model('model/skin.h5',custom_objects={'KerasLayer':hub.KerasLayer})

def sc_dect(image):
    img = tf.io.read_file(image)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [299, 299])
    X_test = np.zeros((1, 299, 299, 3))
    X_test[0]=img
    y_pred = new_model.predict(X_test)
    threshold=0.23
    result = np.zeros((1,))
        # test melanoma probability
    print(y_pred[0][0])

    if y_pred[0][0]<=threshold:
        return 'begnin'

    else:
        print('maligant')
        return 'maligant'

    # if y_pred[0][0] >= threshold:
    #     result[0] = 1
    #     print("malignant")
    #     # return "malignant"
    #     # return "malignant"
    # else:
    #     print("begnin")
    #     return "begnin"
    return "malignant"

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/skin_cancer_detection', methods=['POST', 'GET'])
def skin_cancer_detection():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static/uploads', secure_filename('test.jpeg'))
        print(file_path)
        f.save(file_path)
        result=request.form
        x=sc_dect('static/uploads/test.jpeg')
        print(x)
        if x=='maligant':
            return render_template('skin_cancer_detection.html',result='malignant')
        elif x=='begnin':
            return render_template('skin_cancer_detection.html',result='benign')

    elif request.method=='GET': 
        result=request.form
        return render_template('skin_cancer_detection.html')

@app.route('/help')
def help():
    return render_template('help.html')

        
if __name__ =='__main__':
    app.run(debug=True)
