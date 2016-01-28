from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split

def load_data():
    """Load the Boston dataset."""

    boston = datasets.load_boston()
    return boston

def explore_city_data(city_data):
    """Calculate the Boston housing statistics."""

    # Get the labels and features from the housing data
    housing_prices = city_data.target
    housing_features = city_data.data

    # Size of data (number of houses)?
def size_of_data(city_data):
	number_of_houses = np.shape(city_data.data)[0]
	return number_of_houses
    # Number of features?
def number_of_features(city_data):
	number_of_features = np.shape(city_data.data)[1]
	return number_of_features
    # Minimum price?
def get_min_price(city_data):
	min_price = np.min(city_data.target)
	return min_price
    # Maximum price?
def get_max_price(city_data):
	max_price = np.max(city_data.target)
	return max_price
    # Calculate mean price?
def get_mean_price(city_data):
	mean_price = np.mean(city_data.target)
	return mean_price
    # Calculate median price?
def get_median_price(city_data):
	median_price = np.median(city_data.target)
	return median_price
    # Calculate standard deviation?
def get_standard_deviation(city_data):
	standard_deviation = np.std(city_data.target)
	return standard_deviation



def main():
    """Analyze the Boston housing data. Evaluate and validate the
    performanance of a Decision Tree regressor on the housing data.
    Fine tune the model to make prediction on unseen data."""

    # Load data
    city_data = load_data()

    # Explore the data
    housing_prices = city_data.target
    housing_features = city_data.data
    print np.shape(housing_prices)
    print np.shape(housing_features)

if __name__ == "__main__":
    main()