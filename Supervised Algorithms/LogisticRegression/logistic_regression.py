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
'''

import pandas as pd
import numpy as np

class Config:
    DATA_PATH = "/home/desmond/Desmond/Projects/SimpleNets/Supervised Algorithms/LogisticRegression/data/haberman.data"
    NORMALIZATION = "z_score"
    DATA_SPLIT = 0.8
    LEARNING_RATE = 0.01
    NUM_EPOCHS =  10000

class LogisticRegression():
    def __init__(self) -> None:
        llst_columns = ["Age","Year_Of_Operation", "Positive_Aux_Nodes", "Survival_Status"]
        self.haberman_data = pd.read_csv(Config.DATA_PATH, names=llst_columns)

    def numeric_normalizer(self, pflt_value, pflt_min_value, pflt_max_value, pflt_mean, \
        pflt_std_dev) -> float:
        '''
            This function is used to normalize the numeric data
            We use the scale to range normalization by default ( Scales the values between 0-1 )
        '''
        #-----------------------------------------------------------------------------------
        # Useful when the data is approzimately uniformly distributed
        #-----------------------------------------------------------------------------------
        if Config.NORMALIZATION == "scale_to_range":
            return (pflt_value - pflt_min_value)/(pflt_max_value - pflt_min_value)

        #-----------------------------------------------------------------------------------
        # Useful when the only a few outliers are present in the data
        #-----------------------------------------------------------------------------------
        if Config.NORMALIZATION == "z_score":
            return (pflt_value - pflt_mean)/pflt_std_dev

    def normalize_data(self):
        '''
            From some basic EDA performed we can see that the variables that most
            affect the Survival are Age, haberman_data and Positive_Aux_Nodes
        '''
        lint_min_age = min(self.haberman_data["Age"])
        lint_max_age = max(self.haberman_data["Age"])
        lint_mean_age = self.haberman_data["Age"].mean()
        lint_age_std_dev = self.haberman_data["Age"].std()
        self.haberman_data["Age"] = self.haberman_data["Age"].apply(self.numeric_normalizer, args=[
            lint_min_age, lint_max_age, lint_mean_age, lint_age_std_dev])

        lint_min_year = min(self.haberman_data["Year_Of_Operation"])
        lint_max_year = max(self.haberman_data["Year_Of_Operation"])
        lint_mean_year = self.haberman_data["Year_Of_Operation"].mean()
        lint_year_std_dev = self.haberman_data["Year_Of_Operation"].std()
        self.haberman_data["Year_Of_Operation"] = self.haberman_data["Year_Of_Operation"].apply(self.numeric_normalizer, args=[
            lint_min_year, lint_max_year, lint_mean_year, lint_year_std_dev])

        lint_min_nodes = min(self.haberman_data["Positive_Aux_Nodes"])
        lint_max_nodes = max(self.haberman_data["Positive_Aux_Nodes"])
        lint_mean_nodes = self.haberman_data["Positive_Aux_Nodes"].mean()
        lint_nodes_std_dev = self.haberman_data["Positive_Aux_Nodes"].std()
        self.haberman_data["Positive_Aux_Nodes"] = self.haberman_data["Positive_Aux_Nodes"].apply(self.numeric_normalizer, args=[
            lint_min_nodes, lint_max_nodes, lint_mean_nodes, lint_nodes_std_dev])

        print(self.haberman_data.head(5))
        

    def get_cost(self, train_labels, hypothesis) -> float:
        '''
            Calculate the cost for a particular epoch
        '''
        lint_total_samples = len(train_labels)
        llst_squared_difference = np.power(np.subtract(hypothesis,train_labels), 2)
        return np.sum(llst_squared_difference)/(2*lint_total_samples)

    def get_train_and_test_data(self):
        '''
            Here we split the dataset into train data and test data
        '''
        #-----------------------------------------------------------------------------------
        # Lets convert the normalized dataframe to an np array for faster processing
        #-----------------------------------------------------------------------------------
        normalized_data = np.array(self.haberman_data, dtype=float)

        #-----------------------------------------------------------------------------------
        # Lets add an extra column to for theta_0 (bias). This makes the calculations easier
        #-----------------------------------------------------------------------------------
        normalized_data = np.c_[np.ones(normalized_data.shape[0]), normalized_data]

        lint_partition_index = int(Config.DATA_SPLIT*len(normalized_data))
        
        #-----------------------------------------------------------------------------------
        # For the input features (independent variables) we use [{Bias}, Age, Year_Of_Operation, Positive_Aux_Nodes]
        # For the target (dependent variables) we use the last column [Survival]
        #-----------------------------------------------------------------------------------
        train_features = normalized_data[:lint_partition_index, :4]
        test_features = normalized_data[lint_partition_index:, :4]
        train_labels = normalized_data[:lint_partition_index, -1]
        test_labels = normalized_data[lint_partition_index:, -1]
        
        return train_features, train_labels, test_features, test_labels

    def sigmoid(self, train_features, theta):
        exp_term = np.exp(np.dot(train_features, theta))

        sigm = np.divide(exp_term, np.add(1,exp_term))
        return sigm

    def batch_gradient_ascent(self, theta, train_features, train_labels):
        '''
            Here we iterate n=epoch number of times and call optimize our model parameters
        '''
        llst_cost = []
        for epoch_number in range(Config.NUM_EPOCHS):
            #------------------------------------------------------------------------------
            # Get the h hypothesis/prediction value and calulate the error
            # The formula used is the partial derivative of the cost function
            #------------------------------------------------------------------------------
            hypothesis = self.sigmoid(train_features, theta)
            error = np.subtract(train_labels, hypothesis)
            theta = np.add(theta, Config.LEARNING_RATE * (
                np.dot(np.transpose(train_features), error)/len(train_labels)))

            #------------------------------------------------------------------------------
            # Now we save the actual cost of the epoch for later use/visualization
            #------------------------------------------------------------------------------
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

    def run(self):
        '''
            Main entry point for the program
        '''
        #-----------------------------------------------------------------------------------
        # Normalize data will encode all categories and normalize the numerical data
        #-----------------------------------------------------------------------------------
        self.normalize_data()

        train_features, train_labels, test_features, test_labels = self.get_train_and_test_data()
        
        #-----------------------------------------------------------------------------------
        # The model parameters theta_initial are initialized to randam values
        #-----------------------------------------------------------------------------------
        theta_initial = np.random.rand(train_features.shape[1])

        #-----------------------------------------------------------------------------------
        # Now lets call the batch gradient ascent on our data
        #-----------------------------------------------------------------------------------
        theta_final, llst_cost = self.batch_gradient_ascent(theta_initial, train_features, train_labels)

        _ = self.evaluate_model(theta_final, test_features, test_labels)
        
if __name__ == "__main__":
    lobj_logistic_regression = LogisticRegression()

    '''
        We can set in the following optional parameters in the Config class. 
        Following are the default values set:
            DATA_SPLIT: .8 (80% training set and 20% test set)
            NORMALIZATION: "scale_to_range" (You can set it to z_score as well)
            LEARNING_RATE = 0.01
            NUM_EPOCHS: 10000
        Feel free to tune the hyperparameters the way you want and have fun!
    '''
    lobj_logistic_regression.run()


