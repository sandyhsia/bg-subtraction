import cv2
import numpy as np
cap = cv2.VideoCapture("../../../video_dataset/C.mp4")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

i = 0
while(1):

    if i < 150:
        i += 1
    
    else:

        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 11, 5, 5, 1.5, 2)
        mask1 = np.zeros(np.shape(next))
        #print(np.shape(flow))
        mask1[flow[:, :, 0] > 5] = 255
        mask1[flow[:, :, 1] > 5] = 255
        mask1.astype(np.uint8)
        cv2.imshow('mask1', mask1.astype(np.uint8)) 
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame3', gray)
        (thresh, mask) = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)

        cv2.imshow('frame2',bgr)
        cv2.imshow('frame1', frame2)
        cv2.imshow('mask', mask)

        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',bgr)
        prvs = next

cap.release()
cv2.destroyAllWindows()
