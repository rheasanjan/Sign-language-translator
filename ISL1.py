import cv2
import numpy as np
import util as ut
import svm_train as st
import re
model=st.trainSVM(1) # pass the number of starting character (Eg A-> 1)
#create and train SVM model each time coz bug in opencv 3.1.0 svm.load() https://github.com/Itseez/opencv/issues/4969

# cam=int(input("Enter Camera number: "))
cap=cv2.VideoCapture(0) #camera 0 captures the images

font = cv2.FONT_HERSHEY_SIMPLEX

def nothing(x) :
    pass

label = None

while(cap.isOpened()):
    # cnt = []
    # cv2.waitKey(5000)
    _,img=cap.read()

    cv2.rectangle(img,(300,200),(800,600),(255,0,0),3) # bounding box which captures ISL sign to be detected by the system
    # sleep(2000)


    img1=img[300:600,200:800] #Image within the rectangle is cropped out
    img_ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB) #convert to ycbcr model
    blur = cv2.GaussianBlur(img_ycrcb,(11,11),0)  #the image is blurred
    skin_ycrcb_min = np.array((0, 133, 80)) #trying to detect hand region by giving min and max values
    skin_ycrcb_max = np.array((255, 173, 120))
    mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)  # detecting the hand in the bounding box using skin detection
    contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, 2) #get the contour
    cnt=ut.getMaxContour(contours,4000)	 # using contours to capture the skin filtered image of the hand
    """Variable 'cnt' = maximum contours """
    if cnt is not None:

        gesture,label=ut.getGestureImg(cnt,img1,mask,model)   # passing the trained model for prediction and fetching the result
        print(label) #printing out the predicted label
        """ text to speech """
        """ if previous is equal to present label,then speak"""
        # import pyttsx3
        # engine = pyttsx3.init()
        # engine.say(label)
        # engine.runAndWait()

        # cv2.imshow('PredictedGesture',gesture)				  # showing the best match or prediction
        cv2.putText(img,label,(50,200), font,8,(0,125,155),2)  # displaying the predicted letter on the main screen
        # cv2.putText(img,text,(50,450), font,3,(0,0,255),2)
    cv2.imshow('Frame',img)
    cv2.imshow('Mask',mask)


    k = 0xFF & cv2.waitKey(10)
    if k == 27:
        break
    # del cnt

cap.release()
cv2.destroyAllWindows()
