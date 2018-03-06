# Mega Case Study - Make a Hybrid Deep Learning Model



# Part 1 - Identify the Frauds with the Self-Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
from DataPreProcessing import PreProcess

class AnomalyDetector():
    def __init__(self, path):
        preProcess = PreProcess()
        data = preProcess.pre_processing(path)
        self.X = data
        self.sc = MinMaxScaler(feature_range = (0, 1))
        self.X = self.sc.fit_transform(self.X)
        print (self.X.shape)
        self.som = MiniSom(x = 10, y = 10, input_len = 93, sigma = 1.0, learning_rate = 0.5)
        
    def iterate_fit(self, num_iteration):
        self.som.random_weights_init(self.X)
        self.som.train_random(data = self.X, num_iteration = num_iteration)
        print ("Completed Training")

# Visualizing the results
    def plot_results(self):
        bone()
        pcolor(self.som.distance_map().T)
        # colorbar()
        # for i, x in enumerate(self.X):
        #     w = self.som.winner(x)
        #     plot(w[0] + 0.5,
        #          w[1] + 0.5)
        show()
    def reverse_mapping(self, cooridnates):
        mappings = self.som.win_map(self.X)
        frauds = np.concatenate((mappings[(3,4)], mappings[(8,8)]), axis = 0)
        frauds = self.sc.inverse_transform(frauds)
        possible_frauds = np.where(frauds[:,1] > 500)
        possible_frauds = frauds[possible_frauds,:]
        return possible_frauds


if __name__ == '__main__':
    path = "Transactions.csv"
    anomalyDetector =AnomalyDetector(path)
    anomalyDetector.iterate_fit(100)
    anomalyDetector.plot_results()



# # Part 2 - Going from Unsupervised to Supervised Deep Learning

# # Creating the matrix of features
# customers = dataset.iloc[:, 1:].values

# # Creating the dependent variable
# is_fraud = np.zeros(len(dataset))
# for i in range(len(dataset)):
#     if dataset.iloc[i,0] in frauds:
#         is_fraud[i] = 1

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# customers = sc.fit_transform(customers)


