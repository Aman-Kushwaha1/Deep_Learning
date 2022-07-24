import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0) #device is 0 or filename of video

#classifying face with cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


while True:
    ret, frame = cap.read()

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(frame, 1.1, 5 ) 
        for (x,y,w,h) in face:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 5)   #(x,y), (x,y) format with the width of 5

        #print(frame.shape)
        """plt.imshow(frame)
        plt.show()"""
        cv2.imshow('frame', frame)
    
    if cv2.waitKey(1)== ord('q'):   #press q to quit 
        print("Exiting while loop")
        break

        
cap.release()
cv2.destroyAllWindows()