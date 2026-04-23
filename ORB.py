import cv2
import numpy as np

# file_name = 'image/test1.mp4'
file_name = 'image/test2.mp4'

cap = cv2.VideoCapture(file_name)
cap_delay = cv2.VideoCapture(file_name)


orb = cv2.ORB_create(nfeatures=500,scaleFactor=1.2,nlevels=8,edgeThreshold=31,firstLevel=0,WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE,patchSize=31,fastThreshold=20)
ret, frame = cap.read()
ret_delay, delay_frame = cap_delay.read()
fig = np.hstack((frame,delay_frame))

delay = -5
while True:
    cv2.imshow('camera',fig)
    delay += 1

    ret, frame_new = cap.read()
    if ret:
        frame = frame_new
        kp,des = orb.detectAndCompute(frame,None)
    # image_with_keypoints = cv2.drawKeypoints(frame, kp, None, color=(0,255,0),flags=0)

    if delay>0:
        ret_delay, delay_frame = cap_delay.read()
        if not ret_delay:
            break
        kp_delay,des_delay = orb.detectAndCompute(delay_frame,None)
        # image_delay_with_keypoints = cv2.drawKeypoints(delay_frame, kp_delay, None, color=(0,255,0),flags=0)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        matches = bf.match(des, des_delay)
        # matches = sorted(matches, key = lambda x:x.distance)
        fig = cv2.drawMatches(frame, kp, delay_frame, kp_delay, matches, None, flags=2)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()