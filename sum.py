import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
from sklearn.metrics import accuracy_score
 
def createDataSet():
    filename = 'C:/Users/Zhansaya/Desktop/ML/lab6/data.txt'
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    data = np.zeros((numberOfLines,10))
    fr = open(filename, 'r')
     
    index = 0
    output = open(filename, 'r') # See the r
    listFromLine = output.readlines()
    while index < len(listFromLine):
        array = listFromLine[index].split(',')
        del(array[0])        
        array = [int(i) for i in array]
        data[index,:] = array
        index+=1
         
    return data
 
data = createDataSet()
X = data[:,0:9]
y = data[:,9]
 
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X,y)
 
wrongAns = 0
 
filename = 'C:/Users/Zhansaya/Desktop/ML/lab6/test.txt'
fr = open(filename)
numberOfLines = len(fr.readlines())
data = np.zeros((numberOfLines,10))
fr = open(filename, 'r')
     
predicts = []
origClass = []
index = 0
output = open(filename, 'r') # See the r
listFromLine = output.readlines()
while index < len(listFromLine):
    array = listFromLine[index].split(',')
    del(array[0])        
    array = [int(i) for i in array]
    origClass.append(array[len(array)-1])
    del(array[len(array)-1])
    predicts.append(clf.predict(array))
    index += 1
     
print('Accuracy =', accuracy_score(origClass, predicts))
