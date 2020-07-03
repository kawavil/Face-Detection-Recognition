##Testing face recognizer
import cv2

id = 0
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

nameid = ""
names = []
names_file = open('names.txt', 'r')
names.append('None')
names_text = names_file.read()
for i in names_text.strip().split("\n"):
    person_name = i.strip()
    names.append(person_name)

print(names)
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 700)  # set video widht
cam.set(4, 700)  # set video height
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('model.yml')

font = cv2.FONT_HERSHEY_SIMPLEX

print("opening camera")

results = {}
flag = True
predicted_name = ""
msg1 = ""
msg2 = ""
predicted_names = []


def facerec():
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(int(minW), int(minH)), )

    no_of_faces = len(faces)
    print("No of faces found " + str(no_of_faces))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = face_recognizer.predict(gray[y:y + h, x:x + w])

        # If confidence is less them 100 ==> "0" : perfect match 
        if confidence < 70 and id < len(names):
            name = names[id]
            confidence = "  {0}%".format(round(100 - confidence * 0.3))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            msg1 = "Recognised " + name
            cv2.imwrite("detected_faces/"+ str(name)+ ".jpg", gray[y:y+h,x:x+w])
            print(name)



    flag = False
    cv2.imshow('camera', img)
    k = cv2.waitKey(30) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        print(results)


facerec()
cam.release()
