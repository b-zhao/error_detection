import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest as IF

# isolation forest for finding outliers
class IsolationForest:
    def __init__(self, X, trainSubset = 50, trainCount = 10, threshold = 0.6):
        self.X = X
        print('X shape:', self.X.shape)

        self.trainSubset = trainSubset
        self.trainCount  = trainCount
        self.threshold   = threshold
        self.inlier_X = []
        self.inlier_idx = []
        self.outlier_X = []

    def train(self):
        mask = None
        for i in range(self.trainCount):
            if i % 10 == 0:
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
        self.inlier_idx = bo

    def getInlierX(self):
        return self.inlier_X

    def getOutlierX(self):
        return self.outlier_X
    
    def getInlierIndex(self):
        return self.inlier_idx

'''
# load data
df = pd.read_csv('data/Hospital.csv')
X = np.array(df)
col_names = np.array(df.columns)

# randomly picked two columns for outlier detection
temp = IsolationForest(X[:,[0,5]])
temp.train()
X_outlier = temp.getOutlierX()

print("number of outliers: " + str(np.shape(X_outlier)[0]))
'''