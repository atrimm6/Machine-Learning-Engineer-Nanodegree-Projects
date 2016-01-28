from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split

boston = datasets.load_boston()

def split_data(city_data):
	X, y = city_data.data, city_data.target
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=70)
	return X_train, y_train, X_test, y_test

print np.shape(boston.data)
print np.shape(boston.target)