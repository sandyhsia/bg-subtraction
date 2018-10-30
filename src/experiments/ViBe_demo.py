import VIBE
import numpy as np
import cv2

cap = cv2.VideoCapture("../../../video_dataset/C.mp4")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
print("video has shape:", np.shape(prvs)[0], np.shape(prvs)[1])
r = np.shape(prvs)[0]
c = np.shape(prvs)[1]
#size = (int(c*0.5), int(r*0.5))
#prvs = cv2.resize(prvs, size, interpolation=cv2.INTER_AREA)
vibe = VIBE.ViBe(prvs)

#ret, frame2 = cap.read()
#next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
#fg = vibe.update(next, 2)
#cv2.imshow(fg)

i = 2
while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    #next = cv2.resize(next, size, interpolation=cv2.INTER_AREA)
    fg = vibe.update(next, i)

    cv2.imshow('frame2',fg)
    cv2.imshow('frame1', next)
    combine = np.zeros((r, 2*c))
    combine[:, 0:c] = fg
    combine[:, c:] = next
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',fg)
        cv2.imwrite('opticalgray.png',next)
    #cv2.imwrite(str(i)+'.png', combine)
    prvs = next
    i += 1

cap.release()
cv2.destroyAllWindows()
