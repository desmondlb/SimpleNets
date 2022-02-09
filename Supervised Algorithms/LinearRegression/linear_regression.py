# Lets look at a simple implementaion of linear regression algorithm from scratch
# We use the classic medical insurance dataset

import pandas as pd
import numpy as np

DATA_PATH = "data/insurance.csv"

class LinearRegression():
    def __init__(self) -> None:
        self.ldf_insurance = pd.read_csv(DATA_PATH)

    def numeric_normalizer(self, age, pint_min_value, pint_max_value, pstr_normalization_type):
        
        # We use the scale to range normalization by default ( Scales the values between 0-1 )
        # Useful when the data is approzimately uniformly distributed
        if pstr_normalization_type=="scale_to_range":
            return (age - pint_min_value)/(pint_max_value - pint_min_value)
        
        # Useful when the only a few outliers are present in the data
        if pstr_normalization_type=="z_score":
            # To be implemented
            pass

    def normalize_data(self, normalization_type = "scale_to_range"):
        # Normalize Age
        lint_min_age = min(self.ldf_insurance["age"])
        lint_max_age = max(self.ldf_insurance["age"])
        self.ldf_insurance["age"] = self.ldf_insurance["age"].apply(self.numeric_normalizer, args=[
            lint_min_age, lint_max_age, normalization_type])
        
        # Normalize bmi
        lint_min_bmi = min(self.ldf_insurance["bmi"])
        lint_max_bmi = max(self.ldf_insurance["bmi"])
        self.ldf_insurance["bmi"] = self.ldf_insurance["bmi"].apply(self.numeric_normalizer, args=[
            lint_min_bmi, lint_max_bmi, normalization_type])
        
        # Normalize Children
        lint_min_children = min(self.ldf_insurance["children"])
        lint_max_children = max(self.ldf_insurance["children"])
        self.ldf_insurance["children"] = self.ldf_insurance["children"].apply(
            self.numeric_normalizer, args=[lint_min_children, lint_max_children, normalization_type])
        
        # Categorize sex
        self.ldf_insurance['sex'] = self.ldf_insurance['sex'].astype('category').cat.codes
        # Categorize smoker
        self.ldf_insurance['smoker'] = self.ldf_insurance['smoker'].astype('category').cat.codes
        # Categorize region
        self.ldf_insurance['region'] = self.ldf_insurance['region'].astype('category').cat.codes

    


    def hypothesis(self, inputs, weights):
        return np.matmul(weights, inputs)

    def cost(self, inputs, labels, weights):
        return

    def get_train_and_test_data(self, pflt_train_sample_partition):

        # Lets convert the normalized dataframe to an np array
        normalized_data = np.array(self.ldf_insurance, dtype=float)
        # Get the index at which the data is supposed to be partitioned
        lint_partition_index = int(pflt_train_sample_partition*len(normalized_data))
        
        # For the features (independent variables) we use the 1st 5 columns [age, sex, bmi, children, smoker, region]
        train_features = normalized_data[:lint_partition_index, :5]
        test_features = normalized_data[lint_partition_index:, :5]
        
        # For the labels (dependent variables) we use the last column [charges]
        train_labels = normalized_data[:lint_partition_index, -1]
        test_labels = normalized_data[lint_partition_index:, -1]
        
        return train_features, train_labels, test_features, test_labels

    def gradient_descent(self, theta, bias, train_features, train_labels, epochs, learning_rate):


        # Here we iterate n=epoch number of times and call optimize our model parameters
        for epoch_number in range(epochs):
            # Get the Hypothesis cost and add the bias
            # Insert the formula ithe
            hypothesis = np.sum(np.matmul(theta, train_features), bias)

            # Now we calculate the cost function using the following mse formula
            # Insert the cost funciton formula
            lint_train_samples = len(train_features)
            cost = np.power(np.subtract(hypothesis,train_labels), 2)/lint_train_samples

            # We get the partial derivative of the cost function wrt theta and bias
            # delta_theta = 
            # delta_bias = 
    def run(self, epochs=1000, learning_rate=0.001, train_sample_partition=.65):
        # Now lets call the normalize data which will encode all categories and normalize the numerical data
        self.normalize_data()

        train_features, train_labels, test_features, test_labels = self.get_train_and_test_data(train_sample_partition)
        
        # We generate the initial hypothesis function
        theta, bias = self.generate_hypothesis(train_features.shape, train_labels.shape)

        # Now lets call the gradient descent on our data
        self.gradient_descent(theta, bias, train_features, train_labels, epochs, learning_rate)
        
        

if __name__ == "__main__":
    lobj_linear_regression = LinearRegression()

    # We can pass in the following optional parameters to the run method. Following are the default values
    # epochs: 1000
    # learning_rate (alpha): 0.001
    # train_sample_partition: 65%
    lobj_linear_regression.run()

    # Feel free to tune the hyperparameters the way you want and have fun!

