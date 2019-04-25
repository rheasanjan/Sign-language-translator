import cv2
import numpy as np


# cam=int(raw_input("Enter Camera Index : "))
cap=cv2.VideoCapture(0) #camera
i=12#this variable is used to get the number of the character eg:- A = 1
j=201 #number of the picture. Start with 1
name=""

def nothing(x) :
    pass

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


while(cap.isOpened()): #if camera is opened
    _,img=cap.read() #read the image
    cv2.rectangle(img,(300,200),(800,600),(255,0,0),3) #create the rectangle
    img1=img[300:600,200:800] #just get the image in the rectangle
    img_ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
    blur = cv2.GaussianBlur(img_ycrcb,(11,11),0)
    img_HSV = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
    blur1 = cv2.GaussianBlur(img_HSV,(11,11),0)
    skin_ycrcb_min = np.array((0, 138, 67))
    skin_ycrcb_max = np.array((255, 173, 133))

    mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)
    HSV_mask = cv2.inRange(blur1, (0, 15, 0), (17,170,255))
    contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt=getMaxContour(contours,4000)
    if cnt is not None:
    	x,y,w,h = cv2.boundingRect(cnt)
    	imgT=img1[y:y+h,x:x+w]
    	imgT=cv2.bitwise_and(imgT,imgT,mask=mask[y:y+h,x:x+w])
    	imgT=cv2.resize(imgT,(200,200))
    	cv2.imshow('Trainer',imgT)
    cv2.imshow('Frame',img)
    cv2.imshow('Thresh',mask)
    # cv2.imshow('HSV', HSV_mask)
    k = 0xFF & cv2.waitKey(10)
    if k == 27:
    	break
    if k == 13:
    	name=str(chr(i+64))+"_"+str(j)+".jpg"
    	cv2.imwrite(name,imgT)
    	if(j<400):
    		j+=1
    	else:
    		while(0xFF & cv2.waitKey(0)!=ord('n')):
    			j=201
    		j=201
    		i+=1


cap.release() #release the camera resource
cv2.destroyAllWindows() #close the window
