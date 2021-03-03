import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.optimize
import copy

# For this assignment, assume that every hidden layer has the same number of neurons.
NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = 10
NUM_OUTPUT = 10

# Unpack a list of weights and biases into their individual np.arrays.
def unpack (weightsAndBiases):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN
    W = weightsAndBiases[start:end]
    Ws.append(W)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN*NUM_HIDDEN
        W = weightsAndBiases[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN*NUM_OUTPUT
    W = weightsAndBiases[start:end]
    Ws.append(W)

    Ws[0] = Ws[0].reshape(NUM_HIDDEN, NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN, NUM_HIDDEN)
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN)

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN
    b = weightsAndBiases[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN
        b = weightsAndBiases[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weightsAndBiases[start:end]
    bs.append(b)

    return Ws, bs

def relu(z):
    return np.maximum(0, z)

def relu_d(z):
    ans=z.copy()
    ans[ans<=0]=0
    ans[ans>0]=1
    return ans

def getYHat(zs):
    exp_z = np.exp(zs)
    exp_z_sums = np.sum(exp_z, axis=1)
    y_hat = (exp_z.T/exp_z_sums).T
    return y_hat

def getError(y_te, y_hat):
    no_data = y_te.shape[0]
    err_mat = np.dot(y_te.T, np.log(y_hat))/no_data
    err = -np.mean(err_mat)
    return err

def forward_prop (x, y, weightsAndBiases):
    Ws, bs = unpack(weightsAndBiases)

    zs=np.zeros((NUM_HIDDEN_LAYERS, len(x), NUM_HIDDEN)) 
    hs=[] 
    hs.append(x.T)
    
    for i in range (NUM_HIDDEN_LAYERS - 1):
        zs[i]=(np.dot(Ws[i], hs[-1]).T+bs[i]) ######## Extra Transpose
        hs.append(relu(zs[i]).T)
    yhat=getYHat(zs[-1]) # OR h[-1] ??????
    
    loss = getError(y, yhat)
    # Return loss, pre-activations, post-activations, and predictions
    return loss, zs, hs, yhat
   
def back_prop (x, y, weightsAndBiases, alpha = 0.01):
    loss, zs, hs, yhat = forward_prop(x, y, weightsAndBiases)
    Ws, bs = unpack(weightsAndBiases)

    dJdWs = copy.deepcopy(Ws)  # Gradients w.r.t. weights   # Just for dimentions, deepcopy
    dJdbs = copy.deepcopy(bs)  # Gradients w.r.t. biases    # Just for dimentions, deepcopy

    # TODO    
    # dJdWs[-1] = hs[-1]
    # for i in hs:
    #     print(i.shape)
    g = (yhat - y).T
    for i in range(NUM_HIDDEN_LAYERS -1, -1, -1):  
        # print(i)
        g = g*relu_d(zs[i].T)
        # print(g.shape)
        dJdbs[i] = np.mean(g, axis=1)
        # print((alpha*Ws[i]/len(x)).shape)
        # print(g.shape)
        # print(hs[i-1].shape)
        dJdWs[i] = np.dot(g, hs[i].T) + alpha*Ws[i]/len(x) ##### Regularization 
                                ########## i-1????????????/
        # print(dJdWs[i].shape)
        g = np.dot(Ws[i].T,g)

    # Concatenate gradients
    # print(dJdWs[0].shape)
    # print(Ws[0].shape)
    # print(dJdbs[0].shape)
    # print(bs[0].shape)
    return np.hstack([ dJdW.flatten() for dJdW in dJdWs ] + [ dJdb.flatten() for dJdb in dJdbs ]) 

def train (trainX, trainY, weightsAndBiases, testX, testY):
    NUM_EPOCHS = 100
    trajectory = []
    for epoch in range(NUM_EPOCHS):
        # TODO: implement SGD.
        # TODO: save the current set of weights and biases into trajectory; this is
        # useful for visualizing the SGD trajectory.
        pass
        
    return weightsAndBiases, trajectory

# Performs a standard form of random initialization of weights and biases
def initWeightsAndBiases ():
    Ws = []
    bs = []

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN)
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN)
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])

def plotSGDPath (trainX, trainY, trajectory):
    # TODO: change this toy plot to show a 2-d projection of the weight space
    # along with the associated loss (cross-entropy), plus a superimposed 
    # trajectory across the landscape that was traversed using SGD. Use
    # sklearn.decomposition.PCA's fit_transform and inverse_transform methods.

    def toyFunction (x1, x2):
        return np.sin((2 * x1**2 - x2) / 10.)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(-np.pi, +np.pi, 0.05)  # Just an example
    axis2 = np.arange(-np.pi, +np.pi, 0.05)  # Just an example
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in range(len(axis1)):
        for j in range(len(axis2)):
            Zaxis[i,j] = toyFunction(Xaxis[i,j], Yaxis[i,j])
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

    # Now superimpose a scatter plot showing the weights during SGD.
    Xaxis = 2*np.pi*np.random.random(8) - np.pi  # Just an example
    Yaxis = 2*np.pi*np.random.random(8) - np.pi  # Just an example
    Zaxis = toyFunction(Xaxis, Yaxis)
    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')

    plt.show()


def loadDataset():
    # Load data
    X_tr_raw = (np.load("fashion_mnist_train_images.npy"))
    y_tr_raw = np.load("fashion_mnist_train_labels.npy")
    X_te_raw = (np.load("fashion_mnist_test_images.npy"))
    y_te_raw = np.load("fashion_mnist_test_labels.npy")

    no_data = X_tr_raw.shape[0]

    brightness_value = 256
    X_te = X_te_raw/brightness_value
    X_tr = X_tr_raw/brightness_value
    
    y_tr = np.zeros([X_tr_raw.shape[0], NUM_OUTPUT])
    y_tr_raw = (np.atleast_2d(y_tr_raw).T)
    np.put_along_axis(y_tr, y_tr_raw, 1, axis=1)
    
    y_te = np.zeros([X_te_raw.shape[0], NUM_OUTPUT])
    y_te_raw = (np.atleast_2d(y_te_raw).T)
    np.put_along_axis(y_te, y_te_raw, 1, axis=1)
    return X_te, y_te, X_tr, y_tr


if __name__ == "__main__":
    # TODO: Load data and split into train, validation, test sets
    # trainX = ...
    # trainY = ...
    # ...

    # Initialize weights and biases randomly
    weightsAndBiases = initWeightsAndBiases()
    trainX, trainY, testX, testY = loadDataset()

    # Perform gradient check on random training examples
    print(scipy.optimize.check_grad(lambda wab: forward_prop(np.atleast_2d(trainX[0:5]), np.atleast_2d(trainY[0:5]), wab)[0], \
                                    lambda wab: back_prop(np.atleast_2d(trainX[0:5]), np.atleast_2d(trainY[0:5]), wab), \
                                    weightsAndBiases))

    # weightsAndBiases, trajectory = train(trainX, trainY, weightsAndBiases, testX, testY)
    
    # # Plot the SGD trajectory
    # plotSGDPath(trainX, trainY, ws)
