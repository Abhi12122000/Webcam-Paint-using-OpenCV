#!/usr/bin/env python
# coding: utf-8

# In[34]:


import cv2
import numpy as np
from collections import deque


# In[35]:


blueLow = np.array([100, 60, 60])
blueUp = np.array([140, 255, 255])


# In[36]:


kernel = np.ones((5,5), dtype = np.uint8)


# In[37]:


bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]


# In[38]:


bindex = 0
gindex = 0
rindex = 0


# In[39]:


colors = [(255,0,0),(0,255,0),(0,0,255)]


# In[40]:


colorIndex = 0


# In[41]:


paintWin = np.zeros((600,700,3)) + 255 


# In[42]:


paintWin = cv2.rectangle(paintWin, (40,5), (140,75), (0,0,0), 2)
paintWin = cv2.rectangle(paintWin, (160,5), (255,75), colors[0], -1)
paintWin = cv2.rectangle(paintWin, (275,5), (370,75), colors[1], -1)
paintWin = cv2.rectangle(paintWin, (390,5), (485,75), colors[2], -1)
#paintWin = cv2.rectangle(paintWin, (505,5), (600,75), colors[3], -1)


# In[43]:


paintWin = cv2.putText(paintWin, "CLEAR ALL", (49,33), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,255,255), 2, cv2.LINE_AA)
paintWin = cv2.putText(paintWin, "BLUE", (169,33), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,255,255), 2, cv2.LINE_AA)
paintWin = cv2.putText(paintWin, "GREEN", (284,33), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,255,255), 2, cv2.LINE_AA)
paintWin = cv2.putText(paintWin, "RED", (399,33), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,255,255), 2, cv2.LINE_AA)


# In[44]:


cv2.namedWindow("paint", cv2.WINDOW_AUTOSIZE)

cap = cv2.VideoCapture(0)
while True:
    grabbed, frame = cap.read()
    
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    frame = cv2.rectangle(frame, (40,5), (140,75), (122,122,122), -1)
    frame = cv2.rectangle(frame, (160,5), (255,75), colors[0], -1)
    frame = cv2.rectangle(frame, (275,5), (370,75), colors[1], -1)
    frame = cv2.rectangle(frame, (390,5), (485,75), colors[2], -1)
    
    frame = cv2.putText(frame, "CLEAR ALL", (49,33), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,255,255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "BLUE", (169,33), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,255,255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "GREEN", (284,33), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,255,255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "RED", (399,33), cv2.FONT_HERSHEY_PLAIN, 0.5, (255,255,255), 2, cv2.LINE_AA)
    
    if not grabbed:
        break
    
    blueMask = cv2.inRange(hsv, blueLow, blueUp)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)
    
    contours, _ = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    
    if(len(contours)>0):
        contour = sorted(contours, key = cv2.contourArea, reverse=True)[0]
        
        ((x,y), radius) = cv2.minEnclosingCircle(contour)
        
        cv2.circle(frame, (int(x),int(y)), int(radius), (0,255,255), 1)
        
        M = cv2.moments(contour)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if (center[1] <= 75):
            
            if 40 <= center[0] <= 140:
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                
                bindex = 0
                gindex = 0
                rindex = 0
                
                paintWin[77:,:,:] = 255
                
            elif 160 <= center[0] <= 255:
                colorIndex = 0 
            elif 275 <= center[0] <= 370:
                colorIndex = 1 
            elif 390 <= center[0] <= 485:
                colorIndex = 2
                
        else:
            if colorIndex == 0:
                bpoints[bindex].appendleft(center)
            elif colorIndex == 1:
                gpoints[gindex].appendleft(center)
            elif colorIndex == 2:
                rpoints[rindex].appendleft(center)
    
    else:
        bpoints.append(deque(maxlen=512))
        bindex += 1
        gpoints.append(deque(maxlen=512))
        gindex += 1
        rpoints.append(deque(maxlen=512))
        rindex += 1
        
        points = [bpoints, gpoints, rpoints]
        
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k-1], points[i][j][k], colors[i], 2)
                cv2.line(paintWin, points[i][j][k-1], points[i][j][k], colors[i], 2)
                    
    cv2.imshow("paint", paintWin)
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:




