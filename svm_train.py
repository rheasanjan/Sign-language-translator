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
        self.model = cv2.ml.SVM_create()
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
        # print(hist)
        samples.append(hist)
        # print(samples)
    return np.float32(samples)


#Here goes my wrappers:
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
	for i in range(num+65-1,num+65):
    # for i in range(num+65,num+65+1):

		for j in range(1,401):
			print ('Class '+chr(i)+' is being loaded ')
			imgs.append(cv2.imread('TrainData/'+chr(i)+'_'+str(j)+'.jpg',0))  # all images saved in a list
        # print(imgs)
	labels = np.repeat(np.arange(num,num+1), 400) # label for each corresponding image saved above
	samples=preprocess_hog(imgs)                # images sent for pre processeing using hog which returns features for the images
	print('SVM is building wait some time ...')
	print (len(labels))
	print (len(samples))
	model = SVM(C=2.67, gamma=5.383)
	model.train(samples, labels)  # features trained against the labels using svm
	return model

def predict(model,img):
	samples=hog_single(img)
	resp=model.predict(samples)
	return resp
