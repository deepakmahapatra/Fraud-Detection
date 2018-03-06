# Fraud-Detection
Fraud detection or anomaly detection using self organized maps


The datapreprocessing class is currently configured for the Transaction data in my case.
To apply to any transactional case it should return a dataframe where there is no NAN value 
and each column represents numeric value only.

The AnomalyDetection.py file contains the main method to run the training methods from the minisom class to train 
the self organized maps on the data. The results are plotted on a 2-D map and 
the darker the cells are lower is the distance between two transactions and 
the lighter the colors are the higher is the chances of an anomaly.


