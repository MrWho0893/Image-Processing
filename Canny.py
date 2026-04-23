import cv2;
import numpy as np

cap = cv2.VideoCapture(0)

if cap.isOpened():
    while True:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        Gauss = cv2.GaussianBlur(gray,(7,7),1)
        Canny = cv2.Canny(Gauss,100,200)
        # cv2.imshow('camera',Canny)
        cv2.imshow('camera',Gauss)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()