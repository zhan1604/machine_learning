from __future__ import print_function
import numpy as np
import scipy.io as sio
from sklearn.cross_validation import train_test_split
from pylab import *
from sklearn.metrics import accuracy_score
 
def sigmoid(h):
    return 1 / (1 + exp(-h))
 
def predict(input_layer, theta1, theta2):
    output_layer_size = len(theta2)
 
    input_layer = np.insert(np.array(input_layer),0,1)    
    hidden_layer = np.dot(input_layer, np.transpose(theta1))
    hidden_layer = [sigmoid(i) for i in hidden_layer]
     
    hidden_layer = np.insert(np.array(hidden_layer),0,1)    
    output_layer = np.dot(hidden_layer, np.transpose(theta2))
     
    output = 0
    largest = output_layer[0]
    for i in range(output_layer_size):
        #print(output_layer[i])
        if (largest < output_layer[i]):
            output = i+1
            largest = output_layer[i]
    return output
 
 
mat_contents = sio.loadmat('ex3data1.mat')
#print(mat_contents['X'][1000])
weight_contents = sio.loadmat('ex3weights.mat')
# 0s were converted to 10s in the matlab data because matlab
# indices start at 1, so we need to change them back to 0s
labels = mat_contents['y']
labels = np.where(labels == 10, 0, labels)
labels = labels.reshape((labels.shape[0],))
X = mat_contents['X']
theta1 = weight_contents['Theta1']
theta2 = weight_contents['Theta2']
 
predicted_val = ((zeros(len(X))))
for i in range(len(X)):
    predicted_val[i] = predict(X[i],theta1,theta2)
 
predicted_val = np.where(predicted_val == 10, 0, predicted_val)
print('Accuracy =', accuracy_score(predicted_val, labels))
