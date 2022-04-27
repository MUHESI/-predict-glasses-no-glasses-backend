import time
#import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import load_model
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os

from flask import Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)
tampon = os.path.join(os.getcwd(),"imagesEntrees","datas")
# cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


# API avec JSON
@app.route('/api/predict-glasses', methods=['POST'])
@cross_origin()
def get_reperage():
    raw_images = []
    images = request.files.getlist("img")
    image = images[0]
    image_names = []
    #
    nameFile =  image.filename
    image_names.append(nameFile)
    image.save(os.path.join(tampon, nameFile))
    image.save(os.path.join(os.getcwd(),"data","chargees", nameFile))
    # 
    img_raw = tf.image.decode_image(
            open(tampon+"\\"+nameFile, 'rb').read(), channels=3)
    raw_images.append(img_raw)
    # it is 
    test_datagen = ImageDataGenerator(rescale =1./255)
    image_in = test_datagen.flow_from_directory(os.getcwd()+"\imagesEntrees\\", target_size=(64,64), color_mode='rgb', class_mode='binary', batch_size=1, shuffle=False)

    classOfPrediction=''
    for j in range(len(raw_images)):
        # Reponse en texte
        raw_img = raw_images[j]

        t1 = time.time()
        #Prédiction
        modele= load_model('./model/MonModel.h5')
        predictions = modele.predict(image_in)

        # predictions.reshape()
        t2 = time.time()
        print('time: {}'.format(t2 - t1))

        # if np.argmax(predictions[0]) < 0.5 :
        if predictions[0] < 0.5 :
            classOfPrediction = 'glasses'        
        else :
            classOfPrediction = 'no_glasses'
        # valeur=predictions[0][np.argmax(predictions[0])]
        Confiancevalue = predictions[0]

    print('Repérages:')
    print('JE SUIS {}'.format(predictions[0]))  
    
      # Delete after process
    os.remove(tampon+"\\"+nameFile)
    try:
        return jsonify({
        "success": True,
        'message': "prediction of glasses vs no-glasses done!",
        "NameFile": image_names[0],
        "class": classOfPrediction,
        "confiance": "{}".format(Confiancevalue),
         }), 200

    except FileNotFoundError:
        abort(404)
        
if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=5000)