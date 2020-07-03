##Training Face recognizer 

import cv2
import os
import numpy as np
from skimage import color
import matplotlib.pyplot as plt


def get_detected_faces(cascade, test_image, scaleFactor, minNeighbours):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors= minNeighbours)
    print("No. of faces found : " , len(faces_rect))
    faces = []
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)
        face = gray_image[y:y+h, x:x+w]
        faces.append(face)

    return image_copy, faces

haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def prepare_training_data_from_dataset():
    print("Preparing train data from dataset")
    faces = []
    labels = []
    count = 0

    #i denotes parent folder and j denotes individual image in the folder
    for i in os.listdir('dataset'): 
        count = count +1
        print("label "+ str(count) + " for " + i)
        names_text_file = open("names.txt","a")
        names_text_file.write(i+'\n')
        for j in os.listdir('dataset\\'+i): 
            temp_face = cv2.imread('dataset/'+i+'/'+j, cv2.IMREAD_GRAYSCALE)
            label = count
            faces.append(temp_face)
            labels.append(label)
        names_text_file.close()
    return [faces, labels]


open('names.txt', 'w').close()
[faces, labels] = prepare_training_data_from_dataset()
print( "Faces :" +  str(len(faces)) + "   Labels :" + str(len(labels)))

print("Dataset prepared")
##TRAINING
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
print("Training dataset")
face_recognizer.train(faces, np.array(labels))

print("Training complete")
face_recognizer.write('model.yml') 
