import numpy as np
from flask import Flask, request, jsonify, render_template

import tensorflow as tf
import skimage
import math

app = Flask(__name__)
model = tf.keras.models.load_model('TSR_model.h5')


def prepare_response(classes, error_text='An Error Occured'):
    if classes is not None:
        index = np.argmax(classes)
        if classes[0][index] > 0.6:
            return {
                'class': str(index),
                'class_desc': predictions[index],
                'confidence': math.floor(classes[0][index] * 100)
            }
        else:
            return {
                'class': 'None',
                'class_desc': 'Traffic Sign not present',
                'confidence': math.floor(classes[0][np.argmax(classes)] * 100)
                }
    else:
        return {
                'class': 'Error',
                'class_desc': error_text,
                'confidence': ''
                }


predictions = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }


@app.route("/upload", methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            # print('NO FILE FOUND!')
            return jsonify(prepare_response(None, error_text='Image Upload Error!'))
        file = request.files['file']
        if file.filename == '':
            # print('NO FILE SELECTED')
            return jsonify(prepare_response(None, error_text='File Invalid'))
        if file:
            predict_image = skimage.io.imread(file)
            print('Raw File Shape: ', predict_image.shape)
            if 'png' in file.filename.lower():
                if len(predict_image.shape) <= 2:
                    return jsonify(prepare_response(None, error_text='Image should be either a colored JPG or PNG'))
                predict_image = skimage.color.rgba2rgb(predict_image)
            print('Input File Shape: ', predict_image.shape)
            predict_image128x128 = skimage.transform.resize(predict_image, (128, 128))
            predict_image128x128 = np.array(predict_image128x128)
            print(predict_image128x128.shape)
            predict_image128x128 = np.expand_dims(predict_image128x128, axis=0)
            print(predict_image128x128.shape)
            classes = model.predict(predict_image128x128)
            # print(classes)
            # filename = secure_filename(file.filename)
            # final_path = os.path.join(app.config['UPLOAD_FOLDER'])
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify(prepare_response(classes))
    else:
        return jsonify(prepare_response(None,error_text='Invalid Request'))


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
