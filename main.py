#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from matplotlib import pyplot
import numpy.linalg as la
import MNISTtools
xtrain, ltrain = MNISTtools.load()
print("Shape of xtrain:",xtrain.shape)#Q1
print("Shape of ltrain:",ltrain.shape)#Q1
print("Size of xtrain:",xtrain.size)#Q1
print("Size of ltrain:",ltrain.size)#Q1
print("Dimension of xtrain:",xtrain.ndim)#Q1
print("Dimension of ltrain:",ltrain.ndim)#Q1
print("Image of index 42")
MNISTtools.show(xtrain[:,42])#Q2
print("Label of image of index 42:",ltrain[42])#Q2
print( "Maximum of xtrain:",xtrain.max())
print( "Minimum of xtrain:",xtrain.min())
print("Type of xtrain:",type(xtrain))
x = np.array([[134,224],[200,243],[0,255]])
def normalize_MNIST_images(x):
    x = x.astype(np.float32)
    x = x * (2/255) - 1
    return x
xtrain = normalize_MNIST_images(xtrain)
def label2onehot(lbl):
    d = np.zeros((lbl.max() + 1, lbl.size))
    d[lbl, np.arange(lbl.size)] = 1
    return d
dtrain = label2onehot(ltrain)
def onehot2label(d):
    lbl = d.argmax(axis=0)
    return lbl
def softmax(a):
    y = np.exp(a - a.max(axis = 0)) 
    return y / y.sum(axis = 0)
def softmaxp(a,e):
    #asm = softmax(a)
    #cosang = np.dot(asm,e)
    #sinang = la.norm(np.cross(asm,e))
    #ang = np.arctan2(sinang, cosang)
    #d = np.multiply(asm,e) - ang*asm
    return #d
d = softmax(dtrain)
print(dtrain.shape)
print(d.shape)

