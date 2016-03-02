#import data
import csv
data = []
date = []
symbol = []
i=1
with open("./data/stock_returns_base150.csv", 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        if i==1: 
            i=i+1
            symbol = row
            continue
            
        if row[0]: 
            date.append(row[0])
            data.append(row[1:]) 
            
#convert to numerical data
numCol = len(data[0])
numRow = len(data)
max = [0]* numCol
for i in range(numRow):
    for j in range(numCol):
        if len(data[i][j]) == 0:
            data[i][j] = 0.0
        else:
            data[i][j] = float(data[i][j])
            if abs(data[i][j]) > max[j]:
                max[j] = abs(data[i][j])

#normalize each variable to the same scale [-1,1]
normData = [ [x[i]/max[i] for i in range(numCol)] for x in data]
normAllData = normData
normData[0]



#append S1 data with lags of 1,2,3 as three variables based on model building process
lag = 3
for n in range(lag):
    for i in range(numRow-n-1):
            normData[i+n+1].append(normData[i][0])
dataSet = [x for x in normData if x[0]!=0.0]
predData = [x for x in normData if x[0]==0.0]

#divide data into training set, testing set and prediction set
dataSet = [x for x in normData if x[0]!=0.0]
trainData = dataSet[0:30]
testData = dataSet[30:]

predData = [x for x in normData if x[0]==0.0]

#separate data sets for X(S2-S10) and Y(S1)
trainY = [ y[0] for y in trainData ]
testY = [ y[0] for y in testData ]
trainX = [ y[1:] for y in trainData]
testX = [ y[1:] for y in testData ]


import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

#run the chosen ElasticNet model with alpha of 0.00835
clfcv = linear_model.ElasticNet(alpha = 0.00835)
clfcv.fit(trainX, trainY)       
print("alpha: %d" % clfcv.alpha)
print("coef:" , clfcv.coef_)
print("Residual sum of squares: %.6f"
      % np.mean((clfcv.predict(testX) - testY) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.6f' % clfcv.score(testX, testY))
# Plot outputs
plt.plot(range(len(testY)), clfcv.predict(testX), color='blue',  linewidth=3)
plt.plot(range(len(testY)), testY, color='red',   linewidth=3)
plt.xticks(())
plt.yticks(())

plt.show()

#output prediction for S1
predY = [ y[0] for y in predData ]
predX = [ y[1:] for y in predData]
n= len(predY)
for i in range(n):
    predY[i] = clfcv.predict(predX[i])
    #predY[i] = predY[i].tolist()
    if i+1<n: predX[i+1][-3] = predY[i]
    if i+2<n: predX[i+2][-2] = predY[i]
    if i+3<n: predX[i+3][-1] = predY[i]
        
predYFinal = [x*max[0] for x in predY]

with open('predictions.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
   
    spamwriter.writerow(['Date', 'Value'])
    for i in range(n):
        spamwriter.writerow([date[-n+i], predYFinal[i].tolist()[0]])
    
