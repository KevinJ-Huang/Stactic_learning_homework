import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def relu(y):
    return np.maximum(0.0,y)

def drelu(x):
    tmp = x.copy()
    tmp[tmp >= 0] = 1
    tmp[tmp < 0] = 0
    return tmp

def produce_data():
    X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
     ])
    y = np.array([[0,1,1,0]])
    return X,y

class MLP(object):
    def __init__(self,hidden_num=2,limit=0.01,activation='relu',lr=0.1):
        np.random.seed(8)
        self.X, self.Y = produce_data()
        self.V = np.random.random((self.X.shape[1], hidden_num)) * 2 - 1
        self.W = np.random.random((hidden_num, 1)) * 2 - 1
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.dactivation = dsigmoid
        elif activation == 'relu':
            self.activation = relu
            self.dactivation = drelu
        self.lr = lr
        self.limit = limit
    def update(self):
        L1 = self.activation(np.dot(self.X, self.V))
        L2 =  self.activation(np.dot(L1, self.W))
        L2_delta = (self.Y.T - L2) * self.dactivation(L2)
        L1_delta = L2_delta.dot(self.W.T) * self.dactivation(L1)
        W_C = self.lr * L1.T.dot(L2_delta)
        V_C = self.lr * self.X.T.dot(L1_delta)
        self.W = self.W + W_C
        self.V = self.V + V_C
        return L2

    def train(self):
        L2 = self.update()
        i = 0
        plt.figure()
        error_plot = []
        dot = []
        while(1):
            L2 = self.update()
            i+=1
            error = np.mean(np.abs(self.Y.T-L2))
            if error<self.limit:
                break
            if i%5000==0:
                print('Error:', error)
            if i%500 == 0:
                error_plot.append(error)
                dot.append(i)
        plt.plot(np.array(dot),np.array(error_plot))
        plt.show()
        print("迭代了%.d次收敛"%i)
        # return L2

    def predict(self,data):
        L1 = self.activation(np.dot(data, self.V))
        L2 = self.activation(np.dot(L1, self.W))
        return L2

def plot_decision_boundary(pred_func, X, y, title=None):


    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    for i in range(y.shape[1]):
        plt.scatter(X[i, 0], X[i, 1],c=('y' if y[0][i]>0.5 else 'k'), s=40,  cmap=plt.cm.Spectral)

    if title:
        plt.title(title)
    plt.show()

def tmp(pred):
    sign = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
    ans = sign(nn.predict(pred))
    return ans

X,y = produce_data()

nn = MLP(hidden_num=2,limit=0.01,activation='relu',lr=0.048)
nn.train()
pred = nn.predict(X)
print(pred)
plot_decision_boundary(tmp, X, y, 'Neural Network')
