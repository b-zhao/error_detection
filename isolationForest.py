import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest as IF

# load data
data = pd.read_csv('data/Hospital.csv')
X = np.array(data)
col_names = np.array(data.columns)

# isolation forest for finding outliers
class IsolationForest:
    def __init__(self, X, trainSubset = 50, trainCount  = 10, threshold = 0.6):
        self.X = X
        print('X shape:', self.X.shape)

        self.trainSubset = trainSubset
        self.trainCount  = trainCount
        self.threshold   = threshold
        self.inlier_X = []
        self.outlier_X = []


    def train(self):
        mask = None
        for i in range(self.trainCount):
            print('select a random subset of entries as training data for the', i + 1, 'time')   
            testX = self.X[np.random.choice(self.X.shape[0], self.trainSubset, replace=False), :]
            clf = IF(behaviour='new', contamination='auto')
            clf.fit(testX)
            pred = clf.predict(self.X)
            if mask is None:
                mask = pred 
            else:
                mask = mask + pred 
        threshold = self.threshold
        bo = mask >= threshold * 1 + (1 - threshold) * -1
        self.inlier_X  = self.X[bo]
        self.outlier_X = self.X[bo == False]

    def getInlierX(self):
        return self.inlier_X

    def getOutlierX(self):
        return self.outlier_X


# randomly picked two columns for outlier detection
temp = IsolationForest(X[:,[0,5]])
temp.train()
X_outlier = temp.getOutlierX()

print("number of outliers: " + str(np.shape(X_outlier)[0]))