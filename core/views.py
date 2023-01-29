
import os
import time
import cv2
import numpy as np
import requests
import tensorflow as tf
from bs4 import BeautifulSoup

from rest_framework.response import Response
from rest_framework.views import APIView
from tensorflow import keras

from core.binvis import *
from isphishy import settings



class IsPhishy(APIView):
    
    def predictClass(self, imageFilePath):
        model = keras.models.load_model(r"model//test_model.h5")
        img = cv2.imread(imageFilePath)
        resize = tf.image.resize(img, (256,256))
        yhat = model.predict(np.expand_dims(resize/255, 0))
        
        if yhat > 0.5: 
            return 'Good URL'
        else:
            return 'Bad URL'
        
    def createImage(self, index, htmlFilePath):
        d = open(htmlFilePath, encoding="utf8").read()
        imageFilePath = os.path.join(settings.MEDIA_ROOT, f'{index}.png')
        csource = ColorHilbert(d, None)
        prog = Progress(None)
        drawmap_square("hilbert", 256, csource, imageFilePath, prog)
        os.remove(htmlFilePath)
        return self.predictClass(imageFilePath)
    
    def createHTMLFile(self, index, url):
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        htmlFilePath = os.path.join(settings.MEDIA_ROOT, f'{index}.html')
        htmlFile = open(htmlFilePath, 'w', encoding="utf8")
        htmlFile.write(str(soup))
        htmlFile.close()
        return self.createImage(index, htmlFilePath)
    
    def get(self, request):
        url = request.query_params.get('url')
        index = int(time.time() * 1000000)
        if url is not None:
            predicted_class = self.createHTMLFile(index, url)
            return Response(predicted_class, status = 200)
        return Response("URL toh daal bhai", status = 200)
    