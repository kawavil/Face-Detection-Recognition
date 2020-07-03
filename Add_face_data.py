### Addind individual person's face images in dataset folder 

import cv2
import os
cam = cv2.VideoCapture(0)
#setting video width and height
cam.set(3, 640)
cam.set(4, 480)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# For each person, enter one numeric face id
name = str(input('\n enter user id ==>  '))
print("\n Initializing face capture. Look the camera and wait ...")

count = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)  
        count += 1
        if not os.path.exists("dataset/" + name ):
            os.makedirs("dataset/" + name )
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/" + name + '/' +  str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('face',  gray[y:y+h,x:x+w])
    k = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 500:
        break


print("\n Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
