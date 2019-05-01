import cv2
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import confusion_matrix
from svm_train import SVM

import matplotlib.pyplot as plt
svm_params = dict( kernel_type = cv2.ml.SVM_RBF,
                    svm_type = cv2.ml.SVM_C_SVC,
                    C=2.67, gamma=5.383 )
char_labels = []
class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  #python rapper bug
    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.model = cv2.ml.SVM_create() #create the svm model
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        # print(samples)
        print(responses)
        self.model.train(samples,  cv2.ml.ROW_SAMPLE, responses.astype(int)) # inbuilt training function

    def predict(self, samples):
        # return self.model.predict(samples).ravel()
        tup = self.model.predict(samples)
        # print(tup)
        # nu = np.asarray(tup)
        # print(nu.ravel())
        return tup[1]

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

        samples.append(hist)
    return np.float32(samples)

def hog_single(img):
    samples=[]
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

    samples.append(hist)
    return np.float32(samples)

def trainSVM(num):
    imgs=[]
    for i in range(num+65-1,num+65+25):
    	for j in range(1,326):
    		print ('loading TrainData/'+chr(i)+'_'+str(j)+'.jpg')
    		imgs.append(cv2.imread('TrainData/'+chr(i)+'_'+str(j)+'.jpg',0))
        # print("Loading train data")
    labels = np.repeat(np.arange(num,num+26), 325)
    samples=preprocess_hog(imgs)
    # print(samples)
    print('training SVM...')
    print (len(labels))
    print (len(samples))
    model = SVM(C=2.67, gamma=5.383)
    model.train(samples,labels) #,params=svm_params)
    return model

def testSVM(num):
    imgs=[]
    for i in range(num+65-1,num+65+25):
    	for j in range(326,401):
    		print ('loading TestData/'+chr(i)+'_'+str(j)+'.jpg')
    		imgs.append(cv2.imread('TrainData/'+chr(i)+'_'+str(j)+'.jpg',0))
        # print("Loading Test data...")
    labels_test = np.repeat(np.arange(num,num+26), 75)
    print('testing SVM...')
    print (len(labels_test))
    print (len(imgs))
    return imgs,labels_test

model=trainSVM(1)

test_images,test_labels=testSVM(1)
#print test_labels
count=0.0
k=0
actual_labels = test_labels
predicted_labels = []
for i in test_images:
    test_sample=hog_single(i)
    resp=model.predict(test_sample)
    # print ((int)(resp[0]))
    r = (int)(resp[0])
    predicted_labels.append(r)
    if test_labels[k]==(int)(resp[0]):
    	count+=1.0
    k+=1
# print("predicted=", predicted_labels)
cm = confusion_matrix(actual_labels,predicted_labels, labels=None, sample_weight=None)
print(cm)
print ("accuracy=" , (count/k)*100 ," %")
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
print(cm.diagonal())
fig,ax = plt.subplots()
for i in range(ord('A'), ord('Z')+1):
    char_labels.append(chr(i))
plt.bar(np.arange(1,27),cm.diagonal(),width=0.6)
ax.set_xticks(np.arange(1,27))
ax.set_xticklabels(char_labels)
ax.set_ylabel('Accuracy')
ax.set_xlabel('Characters')

plt.show()
