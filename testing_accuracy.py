import cv2
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import confusion_matrix
from svm_train import SVM
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
        # self.model.setGamma(gamma)
        # self.model.setC(C)
        # self.model.setKernel(cv2.SVM_RBF)
        # self.model.setType(cv2.SVM_C_SVC)

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
    	for j in range(1,151):
    		print ('loading TrainData/'+chr(i)+'_'+str(j)+'.jpg')
    		imgs.append(cv2.imread('TrainData/'+chr(i)+'_'+str(j)+'.jpg',0))
    labels = np.repeat(np.arange(num,num+26), 150)
    samples=preprocess_hog(imgs)
    # print(samples)
    print('training SVM...')
    print (len(labels))
    print (len(samples))
    model = SVM(C=3, gamma=5.383)
    model.train(samples,labels) #,params=svm_params)
    return model

def testSVM(num):
    imgs=[]
    for i in range(num+65-1,num+65+25):
    	for j in range(151,201):
    		print ('loading TestData/'+chr(i)+'_'+str(j)+'.jpg')
    		imgs.append(cv2.imread('TrainData/'+chr(i)+'_'+str(j)+'.jpg',0))
    labels_test = np.repeat(np.arange(num,num+26), 50)
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
print(confusion_matrix(actual_labels,predicted_labels, labels=None, sample_weight=None))
print ("accuracy=" , (count/k)*100 ," %")
