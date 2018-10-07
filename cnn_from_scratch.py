
# coding: utf-8

# In[1]:


import time
import copy

import h5py
import matplotlib.pyplot as plt
import numpy as np

from random import randint
from scipy.signal import convolve2d, correlate2d

plt.style.use('ggplot')


# In[2]:


#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()


# In[3]:


x_train = np.reshape(x_train, (60000, 28, 28))
x_test = np.reshape(x_test, (10000, 28, 28))


# In[4]:


#Implementation of stochastic gradient descent algorithm

#number of inputs
num_inputs = 28

#number of outputs
num_outputs = 10

# Number of filters
num_filters = 4

# Size of filters
k_x = 5
k_y = 5

# Initializing weights and bias using He et al initialization
np.random.seed(10)
model = {}
model['W'] = np.random.randn(num_outputs, num_inputs - k_y + 1, num_inputs - k_x + 1, num_filters) * np.sqrt(2/num_inputs)
model['b'] = np.random.randn(num_outputs) /  np.sqrt(num_inputs)
model['K'] = np.random.randn(k_y, k_x, num_filters) / np.sqrt(num_inputs/2)

model_grads = copy.deepcopy(model)

# helper functions
def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

def activation_function(x):
    return max(x, 0)
#     return np.exp(x) / (1 + np.exp(x))

def activation_derivative(x):
    return np.greater(x, 0).astype(int)
#     return activation_function(x)*(1 - activation_function(x))

def loss_function(f_x, y):
    return -1.0*sum([np.log(f_x[k]) for k in range(num_outputs - 1) if k == y])

def e(y):
    vec = np.zeros(num_outputs)
    vec[y] = 1.0
    return vec

# Convolution function
def convolution(X, K):
    num_inputs = X.shape[0]
    k_x = K.shape[0]
    k_y = K.shape[1]
    x_dim = num_inputs - k_y + 1
    y_dim = num_inputs - k_x + 1
    Z = np.zeros([x_dim, y_dim])
    for i in range(x_dim):
        for j in range(y_dim):
#             print(X[i:i+2, j:j+2])
#             print(K)
            Z[i][j]  = np.multiply(K, X[i:i+k_y, j:j+k_x]).sum()
    return Z


# Using the equations in the notes, all matrices with their usual notation
def forward(x,y, model):

    x_dim = num_inputs - k_y + 1
    y_dim = num_inputs - k_x + 1

    Z = np.zeros([x_dim, y_dim, num_filters])
    H = np.zeros([x_dim, y_dim, num_filters])

    for p in range(num_filters):
        Z[:,:,p] = convolution(x, model['K'][:,:,p])
#     apply activation function over Z
        H[:,:,p] = np.vectorize(activation_function)(Z[:,:,p])
    U = np.tensordot(model['W'], H, axes=((1,2,3),(0,1,2))) + model['b']
    f_x = softmax_function(U)
#     rho = loss_function(f_x, y)
    return Z, H, U, f_x

def backward(x, y, Z, H, U, f_x, model, model_grads):

    dRho_dU = -(e(y) - f_x)
    model_grads['b'] = dRho_dU

    delta = np.tensordot(np.reshape(dRho_dU, (1, num_outputs)), model['W'], axes=((1),(0)))
    delta = np.reshape(delta, (num_inputs - k_y + 1, num_inputs - k_x + 1, num_filters))

    for p in range(num_filters):
        model_grads['K'][:,:,p] = convolution(x, np.multiply(delta, np.vectorize(activation_derivative)(Z))[:,:,p])
#     print(model_grads['b1'].shape)
    model_grads['W'] = np.tensordot(np.reshape(model_grads['b'], (num_outputs, 1)), np.reshape(H, (1, num_inputs - k_y + 1, num_inputs - k_x + 1, num_filters)), axes=((1), (0)))
    return model_grads


# In[6]:


import time
time1 = time.time()
LR = 0.01
num_epochs = 7
for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.005
    if (epochs > 10):
        LR = 0.001
    if (epochs > 15):
        LR = 0.0001
    total_correct = 0
    for n in range(len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
#         Forward step
        Z, H, U, f_x = forward(x, y, model)

        prediction = np.argmax(f_x)
#       Keep track of accuracy
        if (prediction == y):
            total_correct += 1
#       Backward step
        model_grads = backward(x, y, Z, H, U, f_x, model, model_grads)

#       Update model parameters
        model['b'] = model['b'] - LR*model_grads['b']
        model['K'] = model['K'] - LR*model_grads['K']
        model['W'] = model['W'] - LR*model_grads['W']

#     Print accuracy
    print(total_correct/np.float(len(x_train)))
    print(time.time() - time1)
time2 = time.time()
print(time2-time1)
######################################################


# In[11]:


#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    Z, H, U, f_x = forward(x, y, model)
    prediction = np.argmax(f_x)
    if (prediction == y):
        total_correct += 1
print(total_correct/np.float(len(x_test) ) )
