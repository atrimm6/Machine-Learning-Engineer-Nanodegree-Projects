from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
digits = datasets.load_digits()
boston = datasets.load_boston()

print boston.data[0]
print np.shape(boston.data) # print shape of ndarray
print np.shape(boston.data[:,0]) # print shape of column zero
print np.shape(boston.data[0,:]) # print shape of row zero