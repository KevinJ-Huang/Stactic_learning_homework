import numpy as np
from matplotlib import pyplot as plt


def produce():
    data = np.array([[1,2,1],[2,3,1],[3,3,1],[2,1,1],[3,2,1]])
    label = np.array([1,1,1,-1,-1])
    return data,label
    
def sign(w,x):
    res = np.dot(w,x)
    return res

def calulate_w(iters):
    data,label = produce()
    w = np.array([0,0,0])
    for i in range(iters):
        w_old = w
        for j in range(5):
            res = sign(w,data[j])
            if label[j]*res <= 0:
                w = w + label[j]*data[j]
        if (w_old == w).all==True:
            break
    return w

def main():
    data, label = produce()
    w = calulate_w(100)
    print(w)
    X_true = data[:3]
    X_false = data[3:]
    plt.scatter(X_true[:, 0], X_true[:, 1],c='r',label='+')
    plt.scatter(X_false[:, 0], X_false[:, 1], c='b', label='-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Perpetron')
    x1 = np.arange(0,4.0,0.1)
    y1 = (-w[2]-w[0]*x1)/w[1]
    plt.plot(x1,y1)
    plt.legend()
    plt.show()


if __name__=="__main__":
    main()
