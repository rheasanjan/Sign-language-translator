#this is where the model is trained
#SVM is used
import cv2
import numpy as np
from numpy.linalg import norm
svm_params = dict( kernel_type = cv2.ml.SVM_RBF,
                    svm_type = cv2.ml.SVM_C_SVC,
                    C=2.67, gamma=5.383 )
class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  #python rapper bug
    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.model = cv2.ml.SVM_create() #create the svm model
        self.model.setGamma(gamma) #set gamma
        self.model.setC(C) #set regularization parameter
        self.model.setKernel(cv2.ml.SVM_RBF) #radial basis function kernel
        self.model.setType(cv2.ml.SVM_C_SVC) #support vector classifier

    def train(self, samples, responses):
        # print(samples)
        print(responses)
        self.model.train(samples,  cv2.ml.ROW_SAMPLE, responses.astype(int)) # inbuilt training function

    def predict(self, samples):
        tup = self.model.predict(samples) #inbuilt predict function
        return tup[1] #return the predicted label

#feature extraction using HOG
#for multiple images (while training)
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:100,:100], bin[100:,:100], bin[:100,100:], bin[100:,100:]
        mag_cells = mag[:100,:100], mag[100:,:100], mag[:100,100:], mag[100:,100:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        # print(hist)
        samples.append(hist)
        # print(samples)
    return np.float32(samples)


#Here goes my wrappers:
#histogram of oriented gradients for feature extraction
#for a single image (while predicting)
def hog_single(img):
    samples=[]
    # print(np.shape(img))
    # we can deduce that a method to detect edges in an image can be performed
    #by locating pixel locations where the gradient is higher than its neighbors
    #The Sobel Operator is a discrete differentiation operator. It computes an approximation of the gradient of an image intensity function.
    #The Sobel Operator combines Gaussian smoothing and differentiation.
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0,1)#gx is the gradient in x direction
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1,1)#gy is the gradient in y direction
    # cv2.imshow('gx',gx)
    # cv2.imshow('gy',gy)
    mag, ang = cv2.cartToPolar(gx, gy) #get magnitude and angle of the gradient
    bin_n = 16 #no of bins
    # print(np.shape(mag))
    # print(np.shape(ang))
    bin = np.int32(bin_n*ang/(2*np.pi))
    # print(np.shape(bin))
    bin_cells = bin[:100,:100], bin[100:,:100], bin[:100,100:], bin[100:,100:]
    mag_cells = mag[:100,:100], mag[100:,:100], mag[:100,100:], mag[100:,100:]
    print(np.shape(mag_cells))
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    # print(hist)
    # transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= norm(hist) + eps

    samples.append(hist)
    print(np.shape(samples))
    print(samples)
    return np.float32(samples)
#train
def trainSVM(num):
	imgs=[]
    #get the training data (Training data is the form of A_1.jpg)
	for i in range(num+65-1,num+65+25): #change these numbers

		for j in range(1,401):
			print ('Class '+chr(i)+' is loading... ')
			imgs.append(cv2.imread('TrainData/'+chr(i)+'_'+str(j)+'.jpg',0))  # all images saved in a list
	labels = np.repeat(np.arange(num,num+26), 400) # label for each corresponding image saved above. Labels are in the form of integers (Eg: A=1)
	samples=preprocess_hog(imgs) # images sent for pre processeing using hog which returns features for the images
	print('Building SVM...')
	print (len(labels))
	print (len(samples))
	model = SVM(C=2.67, gamma=5.383) #create object of SVM class
	model.train(samples, labels)  # features trained against the labels using svm
	return model

def predict(model,img):
	samples=hog_single(img)
	resp=model.predict(samples)
	return resp
