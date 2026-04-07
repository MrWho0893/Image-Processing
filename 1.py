import cv2
import numpy as np

def Gaussfilter3 (img):
    length = img.shape[0]
    wide = img.shape[1]
    for i in range(length):
        for j in range(wide)
    return GaussFig


cap = cv2.VideoCapture(0)##调用电脑前置摄像头
while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    Gaussfilter3(frame)
    ret,frame = cv2.threshold(frame,180,255,cv2.THRESH_TOZERO_INV)
    cv2.imshow("cam",frame)
    if cv2.waitKey(1) == 27:##如果按下ESC键
        break
cap.release()
cv2.destroyAllWindows()


