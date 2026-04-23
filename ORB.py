import cv2
import numpy as np

# fig1 = cv2.imread()
# fig2 = cv2.imread()
cap = cv2.VideoCapture('./image/538_1776973978.mp4')
cap_delat = cv2.VideoCapture('./image/538_1776973978.mp4')


orb = cv2.ORB_create(nfeatures=500,scaleFactor=1.2,nlevels=8,edgeThreshold=31,firstLevel=0,WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE,patchSize=31,fastThreshold=20)
# kp1,des1 = orb.detectAndCompute(fig1,None)
# kp2,des2 = orb.detectAndCompute(fig2,None)
ret, frame = cap.read()
ret_delat, delat_frame = cap_delat.read()
fig = np.hstack((frame,delat_frame))

a = -10
while True:
    cv2.imshow('camera',fig)
    a += 1

    ret, frame = cap.read()
    kp,des = orb.detectAndCompute(frame,None)
    # image_with_keypoints = cv2.drawKeypoints(frame, kp, None, color=(0,255,0),flags=0)

    if a>0:
        ret_delat, delat_frame = cap_delat.read()
        kp_delat,des_delat = orb.detectAndCompute(delat_frame,None)
        # image_delat_with_keypoints = cv2.drawKeypoints(delat_frame, kp_delat, None, color=(0,255,0),flags=0)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        matches = bf.match(des, des_delat)
        matches = sorted(matches, key = lambda x:x.distance)
        fig = cv2.drawMatches(frame, kp, delat_frame, kp_delat, matches, None, flags=2)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()