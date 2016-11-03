import numpy as np
 
def createDataSet():
    filename = 'C:/Users/Zhansaya/Desktop/ML/kmeans/crime_data.csv'
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    data = np.zeros((numberOfLines,5))
    fr = open(filename, 'r')
     
    index = 1
    output = open(filename, 'r')
    listFromLine = output.readlines()
    while index < len(listFromLine):
        line = listFromLine[index].split(',')
        array = line[1:len(line)]
        array = [float(i) for i in array]
        data[index-1,:] = array
        index+=1
     
    data = data[:index-1,:]
    return data
     
def findClosestCentroids(X, centroids):
    c = np.zeros((len(X)))    
    for i in range(len(X)):
        min_dist = 100000000
        for j in range(len(centroids)):
            dist = 0
            for k in range(len(X[0])):
                dist += (X[i,k] - centroids[j,k])**2
            dist = (dist)**(1/2)
            if dist < min_dist:
                min_dist = dist
                c[i] = j
    return c
 
def computeMeans(X, c, K):
    points_of_c = np.zeros((len(X[0]),K,len(X)))
    cnt = np.zeros((K))
    means = np.zeros((K,len(X[0])))
     
    for i in range(len(X)):
        points_of_c[:,int(c[i]),int(cnt[int(c[i])])] = X[i];
        cnt[int(c[i])] += 1
         
    for i in range(K):
        for j in range(len(means[0])):
            means[i][j] = sum(points_of_c[j][i])/cnt[i]
    return means
     
data = createDataSet()
X = data[:,1:]
y = data[:,0]
y = [int(i-1) for i in y]
 
K = 4
centroids = X[:K,:]
 
iterations = 5
for i in range(iterations):
    idx = findClosestCentroids(X, centroids)
    centroids = computeMeans(X, idx, len(centroids))
 
idx = findClosestCentroids(X, centroids)
s = 0
for cl in range(K):
    cnt = np.zeros((K))
    for i in range(len(idx)):
        if idx[i] == cl:
            cnt[y[i]] += 1
    s += max(cnt)
 
print('Accuracy =', s / len(X))
