import numpy as np
import copy
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
print( "Maximum of xtrain:",xtrain.max())#Q3
print( "Minimum of xtrain:",xtrain.min())#Q3
print("Type of xtrain:",type(xtrain))#Q3
def normalize_MNIST_images(x):#Q4
    x = x.astype(np.float32)
    x = x * (2/255) - 1
    return x
xtrain = normalize_MNIST_images(xtrain)
def label2onehot(lbl):#Q5
    d = np.zeros((lbl.max() + 1, lbl.size))
    d[lbl, np.arange(lbl.size)] = 1
    return d
dtrain = label2onehot(ltrain)
def onehot2label(d):#Q6
    lbl = d.argmax(axis=0)
    return lbl
def softmax(a):#Q7
    y = np.exp(a - a.max(axis = 0)) 
    return y / y.sum(axis = 0)
def softmaxp(a,e):
    asm = softmax(a)
    d = np.multiply(asm,e) - np.sum(asm * e, axis = 0) * asm
    return d
eps = 1e-6 # finite difference step
a = np.random.randn(10, 200) # random inputs
e = np.random.randn(10, 200) # random directions
diff = softmaxp(a, e)
diff_approx = (softmax(a + eps*e) - softmax(a))/eps
rel_error = np.abs(diff - diff_approx).mean() / np.abs(diff_approx).mean()
print(rel_error, 'should be smaller than 1e-6')
def relu(a):
    return np.maximum(a, 0)
def relup(a,e):
    x = copy.deepcopy(a)
    x[x<=0] = 0
    x[x>0] = 1
    return x*e
def init_shallow(Ni, Nh, No):
    b1 = np.random.randn(Nh, 1) / np.sqrt((Ni+1.)/2.)
    W1 = np.random.randn(Nh, Ni) / np.sqrt((Ni+1.)/2.)
    b2 = np.random.randn(No, 1) / np.sqrt((Nh+1.))
    W2 = np.random.randn(No, Nh) / np.sqrt((Nh+1.))
    return W1, b1, W2, b2
Ni = xtrain.shape[0]
Nh = 64
No = dtrain.shape[0]
netinit = init_shallow(Ni, Nh, No)
def forwardprop_shallow(x, net):#Q14
    W1 = net[0]
    b1 = net[1]
    W2 = net[2]
    b2 = net[3]
    a1 = W1.dot(x) + b1
    y = W2.dot(a1) + b2
    return y 
yinit = forwardprop_shallow(xtrain, netinit)
def eval_loss(y, d):
    return -1 * np.sum(d*np.log(np.absolute(y)), axis = 0)
print(eval_loss(yinit, dtrain), 'should be around .26')
ypred = onehot2label(yinit)
print(ypred)
print(ltrain)
def eval_perfs(y, lbl):
    pred = onehot2label(y)
    diff = lbl - pred
    return 1- np.count_nonzero(diff == 0)/diff.size
print(eval_perfs(yinit, ltrain))
def update_shallow(x, d, net, gamma=.05):
    W1 = net[0]
    b1 = net[1]
    W2 = net[2]
    b2 = net[3]
    Ni = W1.shape[1]
    Nh = W1.shape[0]
    No = W2.shape[0]
    gamma = gamma / x.shape[1] # normalized by the training dataset size
    #COMPLETE
    return W1, b1, W2, b2