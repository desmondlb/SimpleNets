'''
    Lets look at a simple implementaion of linear regression algorithm from scratch
    We use the classic medical insurance dataset
    Dataset link: https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv

    Datatype notation :-
    lstr_ : local string variable
    pstr_ : parameter string variable
    lint_ : local integer variable
    pint_ : parameter integer variable
    lflt_ : local floating point variable
    pflt_ : parameter floating point variable

    Reader Comments: ...######...
    Method Description: ''' '''
    Developer Comments: #...
'''

import pandas as pd
import numpy as np

DATA_PATH = "Supervised Algorithms/LinearRegression/data/insurance.csv"

class LinearRegression():
    def __init__(self) -> None:
        self.ldf_insurance = pd.read_csv(DATA_PATH)

    def numeric_normalizer(self, pflt_value, pflt_min_value, pflt_max_value, pstr_normalization_type) -> float:
        '''
            This function is used to normalize the numeric data
            We use the scale to range normalization by default ( Scales the values between 0-1 )
        '''
        ####################################################################################
        # Useful when the data is approzimately uniformly distributed
        ####################################################################################
        if pstr_normalization_type=="scale_to_range":
            return (pflt_value - pflt_min_value)/(pflt_max_value - pflt_min_value)

        ####################################################################################
        # Useful when the only a few outliers are present in the data
        ####################################################################################
        if pstr_normalization_type=="z_score":
            # To be implemented
            pass

    def normalize_data(self, normalization_type = "scale_to_range"):
        '''
            From some basic EDA performed we can see that the variables that most
            affect the insurance charge are Age, BMI and Smoker
        '''
        lint_min_age = min(self.ldf_insurance["age"])
        lint_max_age = max(self.ldf_insurance["age"])
        self.ldf_insurance["age"] = self.ldf_insurance["age"].apply(self.numeric_normalizer, args=[
            lint_min_age, lint_max_age, normalization_type])

        lint_min_bmi = min(self.ldf_insurance["bmi"])
        lint_max_bmi = max(self.ldf_insurance["bmi"])
        self.ldf_insurance["bmi"] = self.ldf_insurance["bmi"].apply(self.numeric_normalizer, args=[
            lint_min_bmi, lint_max_bmi, normalization_type])

        # Normalize the target variable
        # lint_min_charges = min(self.ldf_insurance["charges"])
        # lint_max_charges = max(self.ldf_insurance["charges"])
        # self.ldf_insurance["charges"] = self.ldf_insurance["charges"].apply(self.numeric_normalizer, args=[
        #     lint_min_charges, lint_max_charges, normalization_type])
        
        ####################################################################################
        # Convert smoker to categories as its not a numerical attribute
        ####################################################################################
        self.ldf_insurance['smoker'] = self.ldf_insurance['smoker'].astype('category').cat.codes

    def get_cost(self, train_labels, hypothesis) -> float:
        '''
            Calculate the cost for a particular epoch
        '''
        lint_total_samples = len(train_labels)
        llst_squared_difference = np.power(np.subtract(hypothesis,train_labels), 2)
        return np.sum(llst_squared_difference)/(2*lint_total_samples)

    def get_train_and_test_data(self, pflt_train_sample_partition):
        '''
            Here we split the dataset into train data and test data
        '''
        ####################################################################################
        # Lets convert the normalized dataframe to an np array for faster processing
        ####################################################################################
        normalized_data = self.ldf_insurance.drop(columns=['region', 'children', 'sex'])
        normalized_data = np.array(normalized_data, dtype=float)

        ####################################################################################
        # Lets add an extra column to for theta_0 (bias). This makes the calculations easier
        ####################################################################################
        normalized_data = np.c_[np.ones(normalized_data.shape[0]), normalized_data]

        lint_partition_index = int(pflt_train_sample_partition*len(normalized_data))
        
        ####################################################################################
        # For the input features (independent variables) we use [{Bias}, Age, BMI, Smoker]
        # For the target (dependent variables) we use the last column [charges]
        ####################################################################################
        train_features = normalized_data[:lint_partition_index, :4]
        test_features = normalized_data[lint_partition_index:, :4]
        train_labels = normalized_data[:lint_partition_index, -1]
        test_labels = normalized_data[lint_partition_index:, -1]
        
        return train_features, train_labels, test_features, test_labels

    def batch_gradient_descent(self, theta, train_features, train_labels, epochs, learning_rate):
        '''
            Here we iterate n=epoch number of times and call optimize our model parameters
        '''
        llst_cost = []
        for epoch_number in range(epochs):
            ################################################################################
            # Get the h hypothesis/prediction value and calulate the error
            # The formula used is the partial derivative of the cost function
            ################################################################################
            hypothesis = np.dot(train_features, theta)
            error = np.subtract(hypothesis, train_labels)
            theta = np.subtract(theta, learning_rate * (
                np.dot(np.transpose(train_features), error)/len(train_labels)))

            ################################################################################
            # Now we save the actual cost of the epoch for later use/visualization
            ################################################################################
            llst_cost.append(self.get_cost(train_labels, hypothesis))

        return theta, llst_cost
    
    def evaluate_model(self, theta_final, test_features, test_labels):
        '''
            Check how well the model is performing
        '''
        # To be designed
        llst_comparison = []
        prediction = np.dot(test_features, theta_final)
        for i in range(len(test_labels)):
            llst_comparison.append([prediction[i], test_labels[i]])
        return llst_comparison

    def run(self, epochs=10000, learning_rate=0.01, train_sample_partition=.8):
        '''
            Main entry point for the program
        '''
        ####################################################################################
        # Normalize data will encode all categories and normalize the numerical data
        ####################################################################################
        self.normalize_data()

        train_features, train_labels, test_features, test_labels = self.get_train_and_test_data(
            train_sample_partition)
        
        ####################################################################################
        # The model parameters theta_initial are initialized to randam values
        ####################################################################################
        theta_initial = np.random.rand(train_features.shape[1])

        ####################################################################################
        # Now lets call the batch gradient descent on our data
        ####################################################################################
        theta_final, llst_cost = self.batch_gradient_descent(
            theta_initial, train_features, train_labels, epochs, learning_rate)

        _ = self.evaluate_model(theta_final, test_features, test_labels)
        
if __name__ == "__main__":
    lobj_linear_regression = LinearRegression()

    '''
        We can pass in the following optional parameters to the run method. 
        Following are the default values:
            epochs: 10000
            learning_rate (alpha): 0.01
            train_sample_partition: .8 (80% training set and 20% test set)

        Feel free to tune the hyperparameters the way you want and have fun!
    '''
    lobj_linear_regression.run()


