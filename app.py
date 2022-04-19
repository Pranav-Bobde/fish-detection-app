#Import necessary libraries
from flask import Flask, render_template, request
 
import numpy as np

import tensorflow
import cv2

from keras.preprocessing.image import img_to_array
from keras.applications.resnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.models import load_model
 
#load model
model = load_model("cnn_model_1.h5")
 
print('Model loaded')
 
default_image_size = tuple((256, 256))
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
    
def pred_species(cott_plant):
    testGen =ImageDataGenerator(preprocessing_function= preprocess_input)
    image_list = []
    image_list = (convert_image_to_array(cott_plant))
    np_image_list = np.array(image_list, dtype=np.float16) / 225.0
    np_image_list = tensorflow.reshape(np_image_list, shape=[-1, 256, 256, 3])
    result = model.predict(np_image_list)
    return result
 
#------------>>pred_speciess<<--end
     
 
# Create flask instance
app = Flask(__name__)
 
# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
     

@app.route("/predict", methods=['GET' , 'POST'])
def predict() :
    if request.method == 'POST':
        img = request.files['image']

        img_path = "static/user uploaded" + img.filename    
        img.save(img_path)

        p = pred_species(cott_plant=img_path)
        itemindex = np.where(p==np.max(p))
        list1 = ['Bream','Parkki','Perch','Pike','Roach','Smelt','Whitefish']

    return render_template("index.html", prediction =str(np.max(p)) , fish_species = list1[int(itemindex[1][0])] )
# For local system & cloud


if __name__ == "__main__":
    app.run(debug=True , port=5000)