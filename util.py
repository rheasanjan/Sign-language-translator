import cv2
import numpy as np
import svm_train as st

#Get the biggest Controur
def getMaxContour(contours,minArea=200):
    maxC=None
    maxArea=minArea
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if(area>maxArea):
            maxArea=area
            maxC=cnt
    return maxC


#Get Gesture Image by prediction
#th1 -> segmented image
def getGestureImg(cnt,img,th1,model):
    x,y,w,h = cv2.boundingRect(cnt) #get the height,width of the max contour
    #bounding box around max contour. green rectangle on the screen
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    imgT=img[y:y+h,x:x+w] #get the pixels within the green rectangle
    imgT=cv2.bitwise_and(imgT,imgT,mask=th1[y:y+h,x:x+w])
    # cv2.imshow('bl',imgT)
    imgT=cv2.resize(imgT,(200,200))
    # cv2.imshow('blah',imgT)
    resp=st.predict(model,imgT)
    img=cv2.imread('TrainData/'+chr(int(resp[0])+64)+'_2.jpg')
    return img,chr(int(resp[0])+64)
