import cv2, os
import numpy as np
from PIL import Image 
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
def getImagesWithLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    lds=[]
    for imagePath in imagePaths:
        pillmage=Image.open(imagePath).convert('L')
        imageNp=np.array(pillmage,'uint8')
        ld=int(os.path.split(imagePath)[-1].split(".")[1])
        faces=detector.detectMultiScale(imageNp)
        for(x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            lds.append(ld)
    return faceSamples, lds
faces, lds = getImagesWithLabels('DataSet')
recognizer.train(faces,np.array(lds))
recognizer.save('DataSet/training.xml')
